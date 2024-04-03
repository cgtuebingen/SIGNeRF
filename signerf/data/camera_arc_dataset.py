""" Camera arc dataset module. """

from __future__ import annotations

from typing import Type, List, Dict
from dataclasses import dataclass, field

from rich.console import Console
from torch.utils.data import Dataset

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.configs.base_config import InstantiateConfig

from signerf.utils.poses_generation import circle_poses

CONSOLE = Console(width=120)

@dataclass
class CameraArcDatasetConfig(InstantiateConfig):
    """Camera arc dataset config"""

    _target: Type = field(default_factory=lambda: CameraArcDataset)
    """target class to instantiate"""

    size: int = 10
    """Number of cameras to generate"""
    position: List[float] = field(default_factory=lambda: [0,0,0])
    """Center position of the arc"""
    target: List[float] = field(default_factory=lambda: [0, 0, -0.5])
    """Target position of the arc"""
    radius: float = 1.0
    """Radius of the arc"""
    phi_range: List[float] = field(default_factory=lambda: [0, 324])
    """Start and end angle of rotation about the y-axis in degrees."""
    theta: float = 90.0
    """The angle of elevation from the x-z plane in degrees."""



class CameraArcDataset(Dataset):
    """Dataset that returns camera poses and intrinsics. """

    config: CameraArcDatasetConfig

    def __init__(
        self,
        config: CameraArcDatasetConfig,
        device: str,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
        distortion_params: List[float],
        camera_type: CameraType,
        scale_factor: float = 1,
    ):
        CONSOLE.print(f"Creating CameraArcDataset with config: {config}")
        super().__init__()

        self.device = device
        self.config = config

        self.size = config.size

        self.fx = fx
        self.fy = fy

        self.cx = cx
        self.cy = cy

        self.width = width
        self.height = height

        self.distortion_params = distortion_params
        self.camera_type = camera_type
        self.scale_factor = scale_factor

        self.position = config.position
        self.target = config.target
        self.radius = config.radius
        self.phi_range = config.phi_range
        self.theta = config.theta

        self.cameras = self.create_cameras()
        self.cameras.rescale_output_resolution(scaling_factor=self.scale_factor)

    def __len__(self) -> int:
        return self.size

    def create_cameras(self) -> Cameras:
        """Creates a cameras object from the dataset."""

        # Create camera poses
        poses = circle_poses(
            size=self.size,
            device=self.device,
            radius=self.radius,
            theta=self.theta,
            phi=self.phi_range,
            position=self.position,
            target=self.target,
        )

        # Remove the  0, 0, 0, 1 row
        poses = poses[:, :3, :]

        # Create cameras
        cameras = Cameras(
            camera_to_worlds=poses.detach().cpu(),
            camera_type=self.camera_type,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            width=self.width,
            height=self.height,
            distortion_params=self.distortion_params.detach().cpu(),
        )

        return cameras

    def get_camera(self, camera_idx: int) -> Cameras:
        """Get camera for the given camera index

        Args:
            camera_idx: Camera index

        Returns:
            Cameras: Camera object
        """
        return self.cameras[camera_idx]


    def __getitem__(self, camera_idx: int) -> Dict:
        """ Returns the camera object for the given camera index.

        Args:
            camera_idx: Camera index

        Returns:
            Dict: Camera object
        """

        return self.get_camera(camera_idx)