""" Datamanager for SIGNeRF"""

from __future__ import annotations

from typing import Literal, Dict, Tuple, Type, Union, List, Optional
from dataclasses import dataclass, field

from pathlib import Path
import torch
from pathos.helpers import mp
from torch.nn import Parameter
from rich.progress import Console

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.cameras.cameras import CameraType, Cameras
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datamanagers.parallel_datamanager import DataProcessor
from nerfstudio.data.pixel_samplers import PixelSampler, PixelSamplerConfig
from nerfstudio.data.datamanagers.base_datamanager import DataManager, variable_res_collate, VanillaDataManagerConfig

from signerf.data.signerf_dataparser import SIGNeRFDataParserConfig, SIGNeRFDataParser
from signerf.data.signerf_dataloader import  FixedIndicesEvalCameraDataloader
from signerf.data.camera_arc_dataset import CameraArcDataset
from signerf.data.signerf_patch_pixel_sampler import PatchPixelSamplerConfig

CONSOLE = Console(width=120)

@dataclass
class SIGNeRFDataManagerConfig(VanillaDataManagerConfig):
    """Configuration for the SIGNeRFDataManager."""

    _target: Type = field(default_factory=lambda: SIGNeRFDataManager)
    dataparser: SIGNeRFDataParserConfig = SIGNeRFDataParserConfig()
    """Specifies the dataparser used to unpack the data."""
    num_processes: int = 1
    """Number of processes to use for train data loading. More than 1 doesn't result in that much better performance"""
    queue_size: int = 2
    """Size of shared data queue containing generated ray bundles and batches.
    If queue_size <= 0, the queue size is infinite."""
    max_thread_workers: Optional[int] = None
    """Maximum number of threads to use in thread pool executor. If None, use ThreadPool default."""

class SIGNeRFDataManager(DataManager):
    """Datamanager for SIGNeRF."""

    config: SIGNeRFDataManagerConfig

    def __init__(
        self,
        config: SIGNeRFDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs # pylint: disable=unused-argument
    ):
        self.config: SIGNeRFDataManagerConfig = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"
        self.dataparser_config = self.config.dataparser
        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data
        self.dataparser: SIGNeRFDataParser = self.dataparser_config.setup()
        if test_mode == "inference":
            self.dataparser.downscale_factor = 1  # Avoid opening images
        self.includes_time = self.dataparser.includes_time
        self.train_dataparser_outputs: DataparserOutputs = self.dataparser.get_dataparser_outputs()
        cameras = self.train_dataparser_outputs.cameras
        if len(cameras) > 1:
            for i in range(1, len(cameras)):
                if cameras[0].width != cameras[i].width or cameras[0].height != cameras[i].height:
                    CONSOLE.print("Variable resolution, using variable_res_collate")
                    self.config.collate_fn = variable_res_collate
                    break
        self.train_dataset = self.create_train_dataset()
        # self.eval_dataset = self.create_eval_dataset() # TODO: Implement
        self.exclude_batch_keys_from_device = self.train_dataset.exclude_batch_keys_from_device
        # Spawn is critical for not freezing the program (PyTorch compatability issue)
        # check if spawn is already set
        if mp.get_start_method(allow_none=True) is None:  # type: ignore
            mp.set_start_method("spawn")  # type: ignore
        super().__init__()

    def create_train_dataset(self) -> InputDataset:
        """Sets up the data loaders for training"""
        return InputDataset(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    def create_eval_dataset(self) -> CameraArcDataset | None:
        """Sets up the data loaders for evaluation"""
        #scale_factor=self.config.camera_res_scale_factor,

        # Get the first camera from the training dataset
        camera = self.train_dataset.cameras[0] # pylint: disable=unused-variable

        # Get the first camera from the training dataset
        # eval_dataset = self.config.evalulation_dataset.setup(
        #    device=self.device,
        #    fx = camera.fx.tolist()[0],
        #    fy = camera.fy.tolist()[0],
        #    cx = camera.cx.tolist()[0],
        #    cy = camera.cy.tolist()[0],
        #    width = camera.width.tolist()[0],
        #    height = camera.height.tolist()[0],
        #    distortion_params = camera.distortion_params,
        #    camera_type = camera.camera_type.tolist()[0],
            # Scale factor not needed since we already took care of it in the training dataset
            # scale_factor=self.config.camera_res_scale_factor,
        #)

        # TODO: Implement

        return None

    def _get_pixel_sampler(self, dataset: InputDataset, num_rays_per_batch: int) -> PixelSampler:
        """Infer pixel sampler to use."""
        if self.config.patch_size > 1 and type(self.config.pixel_sampler) is PixelSamplerConfig:
            return PatchPixelSamplerConfig().setup(
                patch_size=self.config.patch_size, num_rays_per_batch=num_rays_per_batch
            )
        is_equirectangular = (dataset.cameras.camera_type == CameraType.EQUIRECTANGULAR.value).all()
        if is_equirectangular.any():
            CONSOLE.print("[bold yellow]Warning: Some cameras are equirectangular, but using default pixel sampler.")

        fisheye_crop_radius = None
        if dataset.cameras.metadata is not None:
            fisheye_crop_radius = dataset.cameras.metadata.get("fisheye_crop_radius")

        return self.config.pixel_sampler.setup(
            is_equirectangular=is_equirectangular,
            num_rays_per_batch=num_rays_per_batch,
            fisheye_crop_radius=fisheye_crop_radius,
        )

    def setup_train(self):
        """Sets up parallel python data processes for training."""
        assert self.train_dataset is not None
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)  # type: ignore
        self.data_queue = mp.Queue(maxsize=self.config.queue_size)
        self.data_procs = [
            DataProcessor(
                out_queue=self.data_queue,
                config=self.config,
                dataparser_outputs=self.train_dataparser_outputs,
                dataset=self.train_dataset,
                pixel_sampler=self.train_pixel_sampler,
            )
            for i in range(self.config.num_processes)
        ]
        for proc in self.data_procs:
            proc.start()
        print("Started threads")

    def setup_eval(self):
        """Sets up the data loader for evaluation"""
        assert self.eval_dataset is not None
        CONSOLE.print("Setting up evaluation dataset...")

        self.fixed_indices_eval_dataloader = FixedIndicesEvalCameraDataloader(
            camera_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the parallel training processes."""
        self.train_count += 1
        bundle, batch = self.data_queue.get()
        ray_bundle = bundle.to(self.device)
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1
        image_batch = next(self.iter_eval_image_dataloader)
        assert self.eval_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.eval_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.eval_ray_generator(ray_indices)
        return ray_bundle, batch

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        for camera, batch in self.eval_dataloader:
            assert camera.shape[0] == 1
            return camera, batch
        raise ValueError("No more eval images")

    def get_train_rays_per_batch(self) -> int:
        """Returns the number of rays per batch for training."""
        if self.train_pixel_sampler is not None:
            return self.train_pixel_sampler.num_rays_per_batch
        return self.config.train_num_rays_per_batch

    def get_eval_rays_per_batch(self) -> int:
        """Returns the number of rays per batch for evaluation."""
        if self.eval_pixel_sampler is not None:
            return self.eval_pixel_sampler.num_rays_per_batch
        return self.config.eval_num_rays_per_batch

    def get_datapath(self) -> Path:
        """Returns the path to the data. This is used to determine where to save camera paths."""
        return self.config.dataparser.data

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        return {}

    def forward(self):
        """Forward pass for the data manager."""

    def __del__(self):
        """Clean up the parallel data processes."""
        if hasattr(self, "data_procs"):
            for proc in self.data_procs:
                proc.terminate()
                proc.join()

