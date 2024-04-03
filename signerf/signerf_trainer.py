""" SIGNeRF Trainer """

from __future__ import annotations

from typing import Dict, Tuple, Type, Union, Literal, Optional
from dataclasses import dataclass, field, asdict

import os
from pathlib import Path
import yaml
import torch
from rich.console import Console

import viser
from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.utils.decorators import check_main_thread
from nerfstudio.utils import profiler, writer
from nerfstudio.engine.callbacks import TrainingCallbackAttributes
from nerfstudio.viewer_legacy.server.viewer_state import ViewerLegacyState

from signerf.interface.interface import Interface
from signerf.interface.viewer import Viewer as ViewerState
from signerf.signerf_pipeline import SIGNeRFPipeline
from signerf.datasetgenerator.datasetgenerator import DatasetGeneratorConfig
from signerf.utils.load_previous_experiment_cameras import load_previous_experiment_cameras


CONSOLE = Console(width=120)

TRAIN_INTERATION_OUTPUT = Tuple[  # pylint: disable=invalid-name
    torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]
]
TORCH_DEVICE = Union[torch.device, str]  # pylint: disable=invalid-name

@dataclass
class SIGNeRFTrainerConfig(TrainerConfig):
    """Configuration for the SIGNeRFTrainer."""
    _target: Type = field(default_factory=lambda: SIGNeRFTrainer)
    """target class to instantiate"""

    reset_optimizer: bool = True
    """whether to reset the optimizer state"""
    reset_scheduler: bool = True
    """whether to reset the scheduler state"""
    reset_step_count: bool = True
    """whether to reset the step count"""
    skip_interface : bool = False
    """whether to skip the interface"""
    skip_generation : bool = False
    """whether to skip the generation (automatically sets skip_interface to True)"""
    previous_experiment_dir: Optional[str] = None
    """path to the previous experiment directory"""


class SIGNeRFTrainer(Trainer):
    """Trainer for SIGNeRF"""

    def __init__(self, config: SIGNeRFTrainerConfig, local_rank: int = 0, world_size: int = 1) -> None:
        self.config = config
        self.ui: Interface | None = None

        # self.training_state: str = "training"

        self.skip_generation = config.skip_generation
        self.skip_interface = config.skip_interface or self.skip_generation
        self.previous_experiment_dir = config.previous_experiment_dir

        if self.previous_experiment_dir is not None and self.skip_generation:
            self.config.pipeline.datamanager.data = self.previous_experiment_dir

            # Save the new config
            self.config.save_config()


        super().__init__(self.config, local_rank, world_size)

    def setup(self, test_mode: Literal["test", "val", "inference"] = "val") -> None:
        """Setup the Trainer by calling other setup functions.

        Args:
            test_mode:
                'val': loads train/val datasets into memory
                'test': loads train/test datasets into memory
                'inference': does not load any dataset into memory
        """

        if self.previous_experiment_dir is not None:
            prev_dataset_generator_config_path = Path(self.previous_experiment_dir) / "config.yml"
            if prev_dataset_generator_config_path.exists():
                prev_dataset_generator_config = yaml.load(prev_dataset_generator_config_path.read_text(), Loader=yaml.Loader)
                assert isinstance(prev_dataset_generator_config, DatasetGeneratorConfig)
                self.config.pipeline.dataset_generator = prev_dataset_generator_config
            else:
                CONSOLE.print("[bold red]Error: [/bold red] Could not find the previous experiment directory, continuing without it")

        self.pipeline: SIGNeRFPipeline = self.config.pipeline.setup(
            device=self.device,
            test_mode=test_mode,
            base_dir = self.config.get_base_dir(),
            world_size=self.world_size,
            local_rank=self.local_rank,
            grad_scaler=self.grad_scaler,
            load_model_with_proposal_weights =  not self.skip_generation,
        )

        self.optimizers = self.setup_optimizers()

        # set up viewer if enabled
        viewer_log_path = self.base_dir / self.config.viewer.relative_log_filename
        self.viewer_state, banner_messages = None, None
        if self.config.is_viewer_legacy_enabled() and self.local_rank == 0:
            CONSOLE.print("Setting up legacy viewer")
            datapath = self.config.data
            if datapath is None:
                datapath = self.base_dir
            self.viewer_state = ViewerLegacyState(
                self.config.viewer,
                log_filename=viewer_log_path,
                datapath=datapath,
                pipeline=self.pipeline,
                trainer=self,
                train_lock=self.train_lock, # pylint: disable=no-member
            )
            banner_messages = [f"Legacy viewer at: {self.viewer_state.viewer_url}"]

        if self.config.is_viewer_enabled() and self.local_rank == 0:
            CONSOLE.print("Setting up viewer")
            datapath = self.config.data
            if datapath is None:
                datapath = self.base_dir
            self.viewer_state = ViewerState(
                self.config.viewer,
                log_filename=viewer_log_path,
                datapath=datapath,
                pipeline=self.pipeline,
                trainer=self,
                train_lock=self.train_lock, # pylint: disable=no-member
                share=self.config.viewer.make_share_url,
            )
            banner_messages = self.viewer_state.viewer_info
        self._check_viewer_warnings()

        self._load_checkpoint()

        self.callbacks = self.pipeline.get_training_callbacks(
            TrainingCallbackAttributes(
                optimizers=self.optimizers, grad_scaler=self.grad_scaler, pipeline=self.pipeline, trainer=self
            )
        )

        # set up writers/profilers if enabled
        writer_log_path = self.base_dir / self.config.logging.relative_log_dir
        writer.setup_event_writer(
            self.config.is_wandb_enabled(),
            self.config.is_tensorboard_enabled(),
            self.config.is_comet_enabled(),
            log_dir=writer_log_path,
            experiment_name=self.config.experiment_name,
            project_name=self.config.project_name,
        )
        writer.setup_local_writer(
            self.config.logging, max_iter=self.config.max_num_iterations, banner_messages=banner_messages
        )
        writer.put_config(name="config", config_dict=asdict(self.config), step=0)
        profiler.setup_profiler(self.config.logging, writer_log_path)

        # Load previous experiment cameras
        reference_camera_to_worlds = None
        synthetic_camera_to_worlds = None
        if self.previous_experiment_dir is not None:
            # Load previous reference cameras
            transform_path = Path(self.previous_experiment_dir) / "transforms.json"
            if transform_path.exists():
                reference_camera_to_worlds, synthetic_camera_to_worlds, is_combined = load_previous_experiment_cameras(transform_path)
                CONSOLE.print("[bold green]Loaded previous experiment cameras[/bold green]")

        # Setup Custom UI
        def rerender_cb():
            if self.ui is not None:
                self.viewer_state._trigger_rerender() # pylint: disable=protected-access

        tabs = self.viewer_state.tabs
        signerf_interface_panel = tabs.add_tab("Generation", viser.Icon.REPLACE)
        with signerf_interface_panel:
            self.ui = Interface(
                self.viewer_state,
                self.pipeline,
                rerender_cb,
                self.exchange_training_dataset,
                trained=self.skip_generation,
                reference_camera_to_worlds=reference_camera_to_worlds,
                synthetic_camera_to_worlds=synthetic_camera_to_worlds,
                previous_experiment_given=self.previous_experiment_dir is not None,
            )

        if not self.skip_generation:
            self.training_state = "paused"
            self.viewer_state.waiting_train.visible = True
            self.viewer_state.pause_train.visible = False

            if not self.skip_interface:
                # Inform the user that the training is paused
                CONSOLE.print("\n[bold yellow]Please continue within the viewer [/bold yellow]")
                CONSOLE.print("You find the generation settings within the 'Generation' tab")
                CONSOLE.print("You can continue training by clicking the 'Generate Dataset & Train' button")
                CONSOLE.print("You can also skip the interface by setting '--skip_interface' to True in the config \n")
                # Logic continues in the interface see handle_training_change

            else:
                CONSOLE.print("[bold yellow]Loading information of generation ... [/bold yellow]")

                if reference_camera_to_worlds is None:
                    CONSOLE.print("[bold red]Error: [/bold red] Could not find the transforms.json file in the previous experiment directory")
                    CONSOLE.print("Starting with interface instead")
                    self.skip_interface = False
                else:
                    # TODO: Don't start viewer as it interferes with the interface

                    self.pipeline.dataset_generator.generate_dataset(
                        graph=self.pipeline.model,
                        reference_camera_to_worlds=reference_camera_to_worlds[:, :3, :],
                        original_dataset=self.pipeline.datamanager.train_dataset if synthetic_camera_to_worlds is None or is_combined else None,
                        synthetic_camera_to_worlds= synthetic_camera_to_worlds[:, :3, :] if synthetic_camera_to_worlds is not None else None,
                        merge_with_original_dataset=is_combined,
                    )

                    # Change the training dataset
                    self.exchange_training_dataset()

                    self.pipeline.load_model_with_proposal_weights = False
                    self.pipeline.reload_model_state_dict_without_proposal_weights()

                    self.viewer_state.waiting_train.visible = False
                    self.viewer_state.pause_train.visible = True
                    self.training_state = "training"


    def exchange_training_dataset(self) -> None:
        """Exchange the training dataset with a new one by reloading the pipeline."""

        pipeline_config = self.pipeline.config
        dataset_generator_config = pipeline_config.dataset_generator
        new_data_path = dataset_generator_config.path / dataset_generator_config.dataset_name

        # Create new pipeline
        pipeline_config.datamanager.data = new_data_path
        new_pipeline = pipeline_config.setup(
            device=self.device,
            base_dir = self.config.get_base_dir(),
            world_size=self.world_size,
            local_rank=self.local_rank,
            grad_scaler=self.grad_scaler,
            load_model_with_proposal_weights = False,
        )

        # Update the pipeline
        self.pipeline = new_pipeline

        # Update the viewer
        self.viewer_state.pipeline = self.pipeline
        self.ui.pipeline = self.pipeline

        # Load the pipeline model
        self._load_checkpoint()

        # Update the callbacks
        self.callbacks = self.pipeline.get_training_callbacks(
            TrainingCallbackAttributes(
                optimizers=self.optimizers, grad_scaler=self.grad_scaler, pipeline=self.pipeline, trainer=self
            )
        )

        self.viewer_state.init_scene(self.pipeline.datamanager.train_dataset, train_state=self.training_state)

        # Save the new config
        self.config.save_config()

    @check_main_thread
    def save_checkpoint(self, step: int) -> None:
        """Save the model and optimizers
        Args:
            step: number of steps in training for given checkpoint
        """
        # possibly make the checkpoint directory
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # save the checkpoint
        ckpt_path = self.checkpoint_dir / f"step-{step:09d}.ckpt"
        pipeline_state_dict = {k: v for k, v in self.pipeline.state_dict().items() if "ip2p." not in k}
        torch.save(
            {
                "step": step,
                "pipeline": self.pipeline.module.state_dict()  # type: ignore
                if hasattr(self.pipeline, "module")
                else pipeline_state_dict,
                "optimizers": {k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()},
                "scalers": self.grad_scaler.state_dict(),
            },
            ckpt_path,
        )
        # possibly delete old checkpoints
        if self.config.save_only_latest_checkpoint:
            # delete everything else in the checkpoint folder
            for f in self.checkpoint_dir.glob("*"):
                if f != ckpt_path:
                    f.unlink()

    def _load_checkpoint(self) -> None:
        """Helper function to load pipeline and optimizer from prespecified checkpoint"""
        load_dir: Path = self.config.load_dir # type: ignore
        if load_dir is not None:
            load_step = self.config.load_step
            if load_step is None:
                print("Loading latest checkpoint from load_dir")
                # NOTE: this is specific to the checkpoint name format
                load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(load_dir))[-1]
            load_path: Path = load_dir / f"step-{load_step:09d}.ckpt"
            assert load_path.exists(), f"Checkpoint {load_path} does not exist"
            loaded_state = torch.load(load_path, map_location="cpu")

            if self.config.reset_step_count:
                self._start_step = 0
            else:
                self._start_step = loaded_state["step"] + 1
            self._start_step = 0

            self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
            if self.config.reset_optimizer:
                CONSOLE.print("Resetting optimizer state")
                self.optimizers = self.setup_optimizers()
            else:
                self.optimizers.load_optimizers(loaded_state["optimizers"])

            if self.config.reset_scheduler:
                CONSOLE.print("Resetting scheduler state")
            else:
                self.grad_scaler.load_state_dict(loaded_state["scalers"])
            CONSOLE.print(f"done loading checkpoint from {load_path}")
        else:
            CONSOLE.print("No checkpoints to load, training from scratch")
