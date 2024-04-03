""" Renderer class to render objects """

import os
import math
from dataclasses import dataclass, field
from typing import List, Type, Union
from pathlib import Path

import torch
import numpy as np
from rich.console import Console

os.environ['PYOPENGL_PLATFORM'] = 'egl'

import trimesh
from pyrender import IntrinsicsCamera, Mesh, Node, OffscreenRenderer, Scene

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.configs import base_config as cfg

CONSOLE = Console(width=120)

@dataclass
class RendererConfig(cfg.InstantiateConfig):
    """Configuration for renderer instantiation"""

    _target: Type = field(default_factory=lambda: Renderer)
    """target class to instantiate"""

    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """position of the object"""
    rotation: List[float] = field(default_factory=lambda: [0, 0, 0])
    """rotation of the object"""
    scale: List[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])
    """scale of the object"""
    color: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    """color of the object"""
    object_path: str = field(default_factory=lambda: "models/bunny.obj")
    """ path to the object """


NERFSTUDIO_BLENDER_SCALE_RATIO: float = 10.0

class Renderer:
    """Renderer class to render objects"""

    def __init__(self, config: RendererConfig, device: str ) -> None:
        self.config = config
        self.device = device

        # Object properties
        self.position = config.position
        self.rotation = config.rotation
        self.scale = config.scale
        self.color = config.color

        self.object_path = config.object_path

        # Scene and mesh
        self.scene = None
        self.mesh = None


    def setup(self) -> None:
        """ Setup the renderer """

        object_path = Path(self.object_path)
        if object_path.suffix != ".obj":
            CONSOLE.print(f"[bold red]Path {object_path} is not an obj file[/bold red]")
            return

        if not object_path.exists():
            CONSOLE.print(f"[bold red]Path {object_path} does not exist[/bold red]")
            CONSOLE.print("Be sure that the path exists on the server not client")
            return

        # Load either Mesh or Scene from file
        loaded = trimesh.load(self.object_path)

        # Convert degrees to radians
        # rotation = [math.radians(-self.rotation[0]), math.radians(-self.rotation[1]), math.radians(self.rotation[2])]
        rotation = [math.radians(self.rotation[0]), math.radians(self.rotation[1]), math.radians(self.rotation[2])]

        # Create rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(rotation[0]), -math.sin(rotation[0])],
            [0, math.sin(rotation[0]), math.cos(rotation[0])]
        ])

        Ry = np.array([
            [math.cos(rotation[1]), 0, math.sin(rotation[1])],
            [0, 1, 0],
            [-math.sin(rotation[1]), 0, math.cos(rotation[1])]
        ])

        Rz = np.array([
            [math.cos(rotation[2]), -math.sin(rotation[2]), 0],
            [math.sin(rotation[2]), math.cos(rotation[2]), 0],
            [0, 0, 1]
        ])

        # Combine rotations into a single rotation matrix
        R = np.dot(Rz, np.dot(Ry, Rx))

        # Create scale matrix
        scaled_scale = [s * NERFSTUDIO_BLENDER_SCALE_RATIO for s in self.scale]
        S = np.diag(scaled_scale)

        # Combine rotation and scale
        RS = np.dot(R, S) #pylint: disable=invalid-name

        # Create full 4x4 transformation matrix
        pose = np.zeros((4, 4))
        pose[0:3, 0:3] = RS
        pose[:, 3] = self.position + [1]  # Append 1 for homogeneous coordinates

        # Convert the scene to a pyrender scene
        if isinstance(loaded, trimesh.Scene):
            self.mesh = Mesh.from_trimesh(loaded.dump(concatenate=True))
        elif isinstance(loaded, trimesh.Trimesh):
            self.mesh = Mesh.from_trimesh(loaded)
        else:
            raise TypeError(f"Unsupported type: {type(loaded)}")

        new_pose = self.convert_matrix_to_pyrender(pose)

        # Scene instance
        light_strength = 1.0
        self.scene = Scene(ambient_light=[light_strength, light_strength, light_strength])
        self.scene.add(self.mesh, pose=new_pose)


    def convert_matrix_to_pyrender(self, matrix):
        """
        Converts a 4x4 matrix to a pyrender matrix (Blender -> OpenGL)

        Args:
            matrix (numpy.ndarray): 4x4 matrix

        Returns:
            numpy.ndarray: 4x4 matrix
        """

        convert = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        return convert @ matrix


    def render_camera(self, camera: Cameras) -> Union[torch.Tensor, torch.Tensor]:
        """
        Render the scene from the camera's perspective

        Args:
            camera (Cameras): camera to render from

        Returns:
            Tensor: rendered image
            Tensor: rendered depth map
            Tensor: rendered normal map
        """

        fx = camera.fx.item()
        fy = camera.fy.item()
        cx = camera.cx.item()
        cy = camera.cy.item()
        width = camera.width.item()
        height = camera.height.item()
        camera_to_worlds = camera.camera_to_worlds.cpu().numpy()

        # Build 4x4 matrix from 4x3
        matrix = np.eye(4)
        matrix[:3, :3] = camera_to_worlds[:3, :3]
        matrix[:3, 3] = camera_to_worlds[:3, 3]

        # Convert matrix to pyrender format
        matrix = self.convert_matrix_to_pyrender(matrix)

        # Renderer instance
        renderer = OffscreenRenderer(viewport_width=width, viewport_height=height)

        # Camera instance
        camera = IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.0001, zfar=10, name="camera")
        camera_node = Node(camera=camera, matrix=matrix)
        self.scene.add_node(camera_node)

        # Render
        color, depth = renderer.render(self.scene)

        # Convert to tensor
        color = torch.from_numpy(color.copy()).to(self.device)
        depth = torch.from_numpy(depth.copy()).unsqueeze(-1).to(self.device)

        # Remove camera
        self.scene.remove_node(camera_node)

        return color, depth
