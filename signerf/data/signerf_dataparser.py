""" SIGNeRF datasetparser """

from __future__ import annotations

from typing import Optional, Type
from dataclasses import dataclass, field

from pathlib import Path, PurePath

import numpy as np
import torch
from PIL import Image
from rich.console import Console
from typing_extensions import Literal

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json

CONSOLE = Console(width=120)
MAX_AUTO_RESOLUTION = 1600


@dataclass
class SIGNeRFDataParserConfig(DataParserConfig):
    """SIGNeRF dataset config"""

    _target: Type = field(default_factory=lambda: SIGNeRFDataParser)
    """target class to instantiate"""
    data: Path = Path()
    """Directory or explicit json file path specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: Optional[int] = None
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "up"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    depth_unit_scale_factor: float = 1e-3
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""


@dataclass
class SIGNeRFDataParser(DataParser):
    """ SIGNeRF dataset parser """

    config: SIGNeRFDataParserConfig
    downscale_factor: Optional[int] = None

    def _generate_dataparser_outputs(self) -> DataparserOutputs:  # pylint: disable=arguments-differ
        """  Generate the dataparser outputs for train

        Returns:
            DataparserOutputs containing data for the specified dataset
        """
        # pylint: disable=too-many-statements

        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."

        if self.config.data.suffix == ".json":
            meta = load_from_json(self.config.data)
            data_dir = self.config.data.parent
        else:
            meta = load_from_json(self.config.data / "transforms.json")
            data_dir = self.config.data

        image_filenames = []
        mask_filenames = []
        depth_filenames = []
        poses = []
        num_skipped_image_filenames = 0

        fx_fixed = "fl_x" in meta
        fy_fixed = "fl_y" in meta
        cx_fixed = "cx" in meta
        cy_fixed = "cy" in meta
        height_fixed = "h" in meta
        width_fixed = "w" in meta
        distort_fixed = False
        for distort_key in ["k1", "k2", "k3", "p1", "p2"]:
            if distort_key in meta:
                distort_fixed = True
                break
        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        original_indices = None
        if "original_indices" in meta:
            original_indices = meta["original_indices"]

        for idx, frame in enumerate(meta["frames"]):
            filepath = PurePath(frame["file_path"])
            fname = self._get_fname(filepath, data_dir)
            if not fname.exists():
                num_skipped_image_filenames += 1
                continue

            if not fx_fixed:
                assert "fl_x" in frame, "fx not specified in frame"
                fx.append(float(frame["fl_x"]))
            if not fy_fixed:
                assert "fl_y" in frame, "fy not specified in frame"
                fy.append(float(frame["fl_y"]))
            if not cx_fixed:
                assert "cx" in frame, "cx not specified in frame"
                cx.append(float(frame["cx"]))
            if not cy_fixed:
                assert "cy" in frame, "cy not specified in frame"
                cy.append(float(frame["cy"]))
            if not height_fixed:
                assert "h" in frame, "height not specified in frame"
                height.append(int(frame["h"]))
            if not width_fixed:
                assert "w" in frame, "width not specified in frame"
                width.append(int(frame["w"]))
            if not distort_fixed:
                distort.append(
                    camera_utils.get_distortion_params(
                        k1=float(frame["k1"]) if "k1" in frame else 0.0,
                        k2=float(frame["k2"]) if "k2" in frame else 0.0,
                        k3=float(frame["k3"]) if "k3" in frame else 0.0,
                        k4=float(frame["k4"]) if "k4" in frame else 0.0,
                        p1=float(frame["p1"]) if "p1" in frame else 0.0,
                        p2=float(frame["p2"]) if "p2" in frame else 0.0,
                    )
                )

            image_filenames.append(fname)

            if "scene_transform_matrix" in frame:
                poses.append(np.array(frame["scene_transform_matrix"]))
            else:
                poses.append(np.array(frame["transform_matrix"]))

            if "_mask_path" in frame: # This is usually mask_path but want to avoid confusion with nerfstudio dataparser
                mask_filepath = PurePath(frame["_mask_path"])
                mask_fname = self._get_fname(
                    mask_filepath,
                    data_dir,
                    downsample_folder_prefix="masks_",
                )

                # Create white mask if frame is not in original_indices
                if original_indices is not None and idx not in original_indices:
                    mask_path = mask_fname.parents[0] / "white.png"

                    # Check if white mask exists
                    if not mask_path.exists():
                        print(f"Creating white mask for frame {idx}")
                        # Create white mask
                        white_mask = Image.new("L", (width[-1], height[-1]), color=255)
                        white_mask.save(mask_path)

                    # Set mask to be completely white
                    mask_filenames.append(mask_path)

                else:
                    mask_filenames.append(mask_fname)

            if "depth_file_path" in frame:
                depth_filepath = PurePath(frame["depth_file_path"])
                depth_fname = self._get_fname(depth_filepath, data_dir, downsample_folder_prefix="depths_")
                depth_filenames.append(depth_fname)

        assert (
            len(image_filenames) != 0
        ), """
        No image files found.
        You should check the file_paths in the transforms.json file to make sure they are correct.
        """
        assert len(mask_filenames) == 0 or (
            len(mask_filenames) == len(image_filenames)
        ), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """
        assert len(depth_filenames) == 0 or (
            len(depth_filenames) == len(image_filenames)
        ), """
        Different number of image and depth filenames.
        You should check that depth_file_path is specified for every frame (or zero frames) in transforms.json.
        """


        # select all images
        num_images = len(image_filenames)
        i_all = np.arange(num_images)
        indices = i_all # Do not split into test and train

        if "orientation_override" in meta:
            orientation_method = meta["orientation_override"]
            CONSOLE.log(f"[yellow] Dataset is overriding orientation method to {orientation_method}")
        else:
            orientation_method = self.config.orientation_method

        poses = torch.from_numpy(np.array(poses).astype(np.float32))

        if "original_transform_matrix" in meta:
            transform_matrix = torch.tensor(meta["original_transform_matrix"], dtype=torch.float32)
        else:
            poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
                poses,
                method=orientation_method,
                center_method=self.config.center_method,
            )


        # Scale poses
        scale_factor = 1.0
        if "original_scale_factor" in meta:
            scale_factor = meta["original_scale_factor"]
        else:
            if self.config.auto_scale_poses:
                scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
            scale_factor *= self.config.scale_factor
            poses[:, :3, 3] *= scale_factor

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
        depth_filenames = [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []
        poses = poses[indices]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        if "camera_model" in meta:
            camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]
        else:
            camera_type = CameraType.PERSPECTIVE

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        fx = torch.tensor(meta["fl_x"], dtype=torch.float32).repeat(len(indices)) if fx_fixed else torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        fy = torch.tensor(meta["fl_y"], dtype=torch.float32).repeat(len(indices)) if fy_fixed else torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        cx = torch.tensor(meta["cx"], dtype=torch.float32).repeat(len(indices)) if cx_fixed else torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        cy = torch.tensor(meta["cy"], dtype=torch.float32).repeat(len(indices)) if cy_fixed else torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        height = torch.tensor(meta["h"], dtype=torch.int32).repeat(len(indices)) if height_fixed else torch.tensor(height, dtype=torch.int32)[idx_tensor]
        width = torch.tensor(meta["w"], dtype=torch.int32).repeat(len(indices)) if width_fixed else torch.tensor(width, dtype=torch.int32)[idx_tensor]
        if distort_fixed:
            distortion_params = camera_utils.get_distortion_params(
                k1=float(meta["k1"]) if "k1" in meta else 0.0,
                k2=float(meta["k2"]) if "k2" in meta else 0.0,
                k3=float(meta["k3"]) if "k3" in meta else 0.0,
                k4=float(meta["k4"]) if "k4" in meta else 0.0,
                p1=float(meta["p1"]) if "p1" in meta else 0.0,
                p2=float(meta["p2"]) if "p2" in meta else 0.0,
            )
            distortion_params = distortion_params.repeat(len(indices)).reshape([len(indices), distortion_params.shape[0]])
        else:
            distortion_params = torch.stack(distort, dim=0)[idx_tensor]

        camera_to_worlds = poses[:, :3, :4]

        # Remove masks in case of non merging dataset
        if "original_indices" not in meta:
            mask_filenames = []


        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=camera_to_worlds,
            camera_type=camera_type,
        )

        assert self.downscale_factor is not None
        cameras.rescale_output_resolution(scaling_factor=1.0 / self.downscale_factor)

        if "applied_transform" in meta:
            applied_transform = torch.tensor(meta["applied_transform"], dtype=transform_matrix.dtype)
            transform_matrix = transform_matrix @ torch.cat(
                [applied_transform, torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype)], 0
            )
        if "applied_scale" in meta:
            applied_scale = float(meta["applied_scale"])
            scale_factor *= applied_scale

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata={
                "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor
            },
        )
        return dataparser_outputs

    def get_dataparser_outputs(self) -> DataparserOutputs: # pylint: disable=arguments-differ
        """Returns the dataparser outputs for the given split.

        Args:
            split: Which dataset split to generate (train/test).

        Returns:
            DataparserOutputs containing data for the specified dataset and split
        """
        dataparser_outputs = self._generate_dataparser_outputs()
        return dataparser_outputs


    def _get_fname(self, filepath: PurePath, data_dir: PurePath, downsample_folder_prefix="images_") -> Path:
        """Get the filename of the image file.
        downsample_folder_prefix can be used to point to auxiliary image data, e.g. masks

        filepath: the base file name of the transformations.
        data_dir: the directory of the data that contains the transform file
        downsample_folder_prefix: prefix of the newly generated downsampled images
        """

        if self.downscale_factor is None:
            if self.config.downscale_factor is None:
                test_img = Image.open(data_dir / filepath)
                h, w = test_img.size
                max_res = max(h, w)
                df = 0
                while True:
                    if (max_res / 2 ** (df)) < MAX_AUTO_RESOLUTION:
                        break
                    if not (data_dir / f"{downsample_folder_prefix}{2**(df+1)}" / filepath.name).exists():
                        break
                    df += 1

                self.downscale_factor = 2**df
                CONSOLE.log(f"Auto image downscale factor of {self.downscale_factor}")
            else:
                self.downscale_factor = self.config.downscale_factor

        if self.downscale_factor > 1:
            return data_dir / f"{downsample_folder_prefix}{self.downscale_factor}" / filepath.name
        return data_dir / filepath
