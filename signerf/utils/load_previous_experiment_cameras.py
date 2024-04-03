""" Load the previous experiment cameras from the transforms file """

import json

from pathlib import Path
from typing import Tuple, List, Union

import torch
from torch import Tensor


def load_previous_experiment_cameras(transforms_path: Path) -> Tuple[Tensor, Union[Tensor, None], bool]:
    """Load the previous experiment cameras from the transforms file

    Args:
        transforms_path (Path): Path to the transforms file

    Returns:
        Tuple[Tensor, Tensor]: Tuple of the reference camera to worlds and synthetic camera to worlds
    """

    with open(transforms_path) as f: # pylint: disable=unspecified-encoding
        transforms = json.load(f)

    reference_camera_to_worlds_list: List[Tensor] = []
    reference_indices = transforms["reference_indices"]

    frames = transforms["frames"]
    reference_frames = [frames[i] for i in reference_indices]

    for i, frame in enumerate(reference_frames):
        transform_matrix = frame["scene_transform_matrix"]
        reference_camera_to_worlds_list.append(torch.tensor(transform_matrix[:3], dtype=torch.float32))

    reference_camera_to_worlds: Tensor = torch.stack(reference_camera_to_worlds_list, dim=0)

    if "is_synthetic" in transforms and transforms["is_synthetic"]:
        synthetic_camera_to_worlds_list: List[Tensor] = []
        synthetic_indices = transforms["generated_indices"]
        synthetic_frames = [frames[i] for i in synthetic_indices]

        for i, frame in enumerate(synthetic_frames):
            transform_matrix = frame["scene_transform_matrix"]
            synthetic_camera_to_worlds_list.append(torch.tensor(transform_matrix[:3], dtype=torch.float32))

        synthetic_camera_to_worlds: Tensor = torch.stack(synthetic_camera_to_worlds_list, dim=0)
    else:
        synthetic_camera_to_worlds = None

    is_combined = False
    if "is_combined" in transforms:
        is_combined = transforms["is_combined"]

    return reference_camera_to_worlds, synthetic_camera_to_worlds, is_combined
