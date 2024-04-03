"""SIGNeRF pipeline"""

from dataclasses import dataclass, field

from pathlib import Path
from typing import Any, List, Mapping, Optional, Type
from typing_extensions import Literal
from rich.console import Console

from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig

from signerf.data.signerf_datamanager import SIGNeRFDataManagerConfig
from signerf.datasetgenerator.datasetgenerator import DatasetGenerator, DatasetGeneratorConfig

CONSOLE = Console(width=120)

@dataclass
class SIGNeRFPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: SIGNeRFPipeline)
    """target class to instantiate"""
    datamanager: SIGNeRFDataManagerConfig = field(default_factory=lambda: SIGNeRFDataManagerConfig)
    """specifies the datamanager config"""
    dataset_generator: DatasetGeneratorConfig = field(default_factory=lambda: DatasetGeneratorConfig)
    """specifies the dataset generator config"""

class SIGNeRFPipeline(VanillaPipeline):
    """SIGNeRF pipeline"""

    config: SIGNeRFPipelineConfig

    def __init__(
        self,
        config: SIGNeRFPipelineConfig,
        device: str,
        base_dir: Path = None, #pylint: disable=unused-argument
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None, #pylint: disable=unused-argument
        load_model_with_proposal_weights: bool = True,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank)


        dataset_generator_config = config.dataset_generator

        self.dataset_generator: DatasetGenerator = dataset_generator_config.setup(
            original_transform_matrix=self.datamanager.train_dataparser_outputs.dataparser_transform,
            original_scale_factor=self.datamanager.train_dataparser_outputs.dataparser_scale,
            transform_poses_to_original_space=self.datamanager.train_dataparser_outputs.transform_poses_to_original_space,
            device=device
        )

        # Set the camera parameters if not set
        if self.dataset_generator.fx is None:
            first_camera_fx = self.datamanager.train_dataset.cameras.fx[0].item()
            self.dataset_generator.fx = first_camera_fx
            self.dataset_generator.config.fx = first_camera_fx

        if self.dataset_generator.fy is None:
            first_camera_fy = self.datamanager.train_dataset.cameras.fy[0].item()
            self.dataset_generator.fy = first_camera_fy
            self.dataset_generator.config.fy = first_camera_fy

        if self.dataset_generator.cx is None:
            first_camera_cx = self.datamanager.train_dataset.cameras.cx[0].item()
            self.dataset_generator.cx = first_camera_cx
            self.dataset_generator.config.cx = first_camera_cx

        if self.dataset_generator.cy is None:
            first_camera_cy = self.datamanager.train_dataset.cameras.cy[0].item()
            self.dataset_generator.cy = first_camera_cy
            self.dataset_generator.config.cy = first_camera_cy

        if self.dataset_generator.width is None:
            first_camera_width = self.datamanager.train_dataset.cameras.width[0].item()
            self.dataset_generator.width = first_camera_width
            self.dataset_generator.config.width = first_camera_width

        if self.dataset_generator.height is None:
            first_camera_height = self.datamanager.train_dataset.cameras.height[0].item()
            self.dataset_generator.height = first_camera_height
            self.dataset_generator.config.height = first_camera_height

        self.load_model_with_proposal_weights = load_model_with_proposal_weights
        self.model_state_dict = None

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        is_ddp_model_state = True
        model_state = {}
        for key, value in state_dict.items():
            if key.startswith("_model."):
                # remove the "_model." prefix from key
                model_state[key[len("_model.") :]] = value
                # make sure that the "module." prefix comes from DDP,
                # rather than an attribute of the model named "module"
                if not key.startswith("_model.module."):
                    is_ddp_model_state = False
        # remove "module." prefix added by DDP
        if is_ddp_model_state:
            model_state = {key[len("module.") :]: value for key, value in model_state.items()}

        pipeline_state = {key: value for key, value in state_dict.items() if not key.startswith("_model.")}

        if "field.embedding_appearance.embedding.weight" in model_state:
            del model_state["field.embedding_appearance.embedding.weight"]

        if "datamanager.train_camera_optimizer.pose_adjustment" in pipeline_state:
            del pipeline_state["datamanager.train_camera_optimizer.pose_adjustment"]

        if "datamanager.train_ray_generator.pose_optimizer.pose_adjustment" in pipeline_state:
            del pipeline_state["datamanager.train_ray_generator.pose_optimizer.pose_adjustment"]

        # TODO: Maybe merge weights with the original cameras
        if "camera_optimizer.pose_adjustment" in model_state:
            del model_state["camera_optimizer.pose_adjustment"]

        self.model_state_dict = model_state

        # Delete proposal weights
        if not self.load_model_with_proposal_weights:
            for key in list(model_state.keys()):
                if key.startswith("proposal"):
                    del model_state[key]

        self.model.load_state_dict(model_state, strict=False)
        super().load_state_dict(pipeline_state, strict=False)


    def reload_model_state_dict_without_proposal_weights(self):
        """Reloads the model state dict without proposal weights"""

        CONSOLE.print("Reloading model without proposal weights")
        if self.model_state_dict is not None:
            model_state_dict = self.model_state_dict
            for key in list(model_state_dict.keys()):
                if key.startswith("proposal"):
                    del model_state_dict[key]
            self.model.load_state_dict(model_state_dict, strict=False)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks from both the Dataloader and the Model."""
        datamanager_callbacks = self.datamanager.get_training_callbacks(training_callback_attributes)
        model_callbacks = self.model.get_training_callbacks(training_callback_attributes)
        callbacks = datamanager_callbacks + model_callbacks

        return callbacks

    def forward(self):
        pass
