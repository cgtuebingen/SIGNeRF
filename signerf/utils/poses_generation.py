""" This module contains utility functions for generating camera poses. """

from typing import Tuple, List

import math
import torch


def safe_normalize(x: torch.Tensor, eps=1e-20) -> torch.Tensor:
    """ Normalizes the input tensor x to have unit norm.

    Args:
        x (torch.Tensor): The input tensor.
        eps (float): A small value to avoid division by zero.

    Returns:
        torch.Tensor: The normalized tensor.
    """

    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

def circle_poses(
    size: int,
    device: torch.device,
    radius: float,
    theta: float,
    phi: Tuple[float, float],
    position: List[float],
    target: List[float],
) -> torch.Tensor:
    """
    Computes the camera pose for a circle around the object of interest.

    Args:
        size (int): The number of poses to generate.
        device (torch.device): The device on which to create the poses.
        radius (float): The radius of the circle.
        theta (float): The angle of elevation from the x-z plane in degrees.
        phi (float): The angle of rotation about the y-axis in degrees.
        position (List[float]): The center position of the circle.
        target (List[float]): The target position to look at.

    Returns:
        torch.Tensor: A tensor of shape (size, 4, 4) representing the camera poses.
    """

    # Coordsystem: x forward, y right, z up

    # Convert angles to radians
    theta = math.radians(theta)
    phi_start = math.radians(phi[0])
    phi_end = math.radians(phi[1])

    poses = torch.eye(4, dtype=torch.float, device=device).repeat(size, 1, 1)
    phis = torch.linspace(phi_start, phi_end, size, device=device)
    theta = torch.FloatTensor([theta]).to(device)

    # Compute the camera position
    poses[:, 0, 3] = radius * torch.sin(theta) * torch.cos(phis) + position[0] # X
    poses[:, 1, 3] = radius * torch.sin(theta) * torch.sin(phis) + position[1] # Y
    poses[:, 2, 3] = radius * torch.cos(theta) + position[2]                   # Z


    # Calculate the camera orientation to look at the target
    z = safe_normalize(poses[:, :3, 3] - torch.FloatTensor(target).to(device))
    x = safe_normalize(torch.cross(torch.FloatTensor([0, 0, 1]).to(device).repeat(size, 1), z))
    y = safe_normalize(torch.cross(z, x))

    poses[:, :3, 0] = x
    poses[:, :3, 1] = y
    poses[:, :3, 2] = z

    return poses


def random_sphere_poses(
    size: int,
    device: torch.device,
    radius: float,
    theta: Tuple[float, float],
    phi: Tuple[float, float],
    position: List[float],
    target: List[float],
) -> torch.Tensor:
    """
    Computes the camera pose for a random sphere around the object of interest.

    Args:
        size (int): The number of poses to generate.
        device (torch.device): The device on which to create the poses.
        radius (float): The radius of the sphere.
        theta (float): Theta angle range in degrees.
        phi (float): Phi angle range in degrees.
        position (List[float]): The center position of the sphere.
        target (List[float]): The target position to look at.

    Returns:
        torch.Tensor: A tensor of shape (size, 4, 4) representing the camera poses.
    """

    # Coordsystem: x forward, y right, z up

    # Convert angles to radians
    theta_start = math.radians(theta[0])
    theta_end = math.radians(theta[1])
    phi_start = math.radians(phi[0])
    phi_end = math.radians(phi[1])

    poses = torch.eye(4, dtype=torch.float, device=device).repeat(size, 1, 1)
    thetas = torch.FloatTensor([theta]).to(device)

    # Randomly sample the angles
    theta_min = (1 - math.cos(theta_start)) * 0.5
    theta_max = (1 - math.cos(theta_end)) * 0.5
    thetas = torch.rand(size, device=device) * (theta_max - theta_min) + theta_min
    thetas = torch.acos(1 - 2 * thetas)

    phis = torch.rand(size, device=device) * (phi_end - phi_start) + phi_start

    # Compute the camera position with spherical to Cartesian conversion
    poses[:, 0, 3] = radius * torch.sin(thetas) * torch.cos(phis) + position[0] # X
    poses[:, 1, 3] = radius * torch.sin(thetas) * torch.sin(phis) + position[1] # Y
    poses[:, 2, 3] = radius * torch.cos(thetas) + position[2]                   # Z

    # Calculate the camera orientation to look at the target
    z = safe_normalize(poses[:, :3, 3] - torch.FloatTensor(target).to(device))
    x = safe_normalize(torch.cross(torch.FloatTensor([0, 0, 1]).to(device).repeat(size, 1), z))
    y = safe_normalize(torch.cross(z, x))

    poses[:, :3, 0] = x
    poses[:, :3, 1] = y
    poses[:, :3, 2] = z

    return poses
