""" Dataset generator class to generate a dataset with the given parameters """

from __future__ import annotations
from typing import Literal, Type, Tuple, Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field

import datetime
from pathlib import Path
import json
import time
import math
import yaml
import torch
from PIL import Image
import cv2
from torch import Tensor
import torch.nn.functional as F
from rich.progress import Console, BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from nerfstudio.configs import base_config as cfg
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.base_model import Model
from nerfstudio.data.datasets.base_dataset import InputDataset

from signerf.renderer.renderer import Renderer, RendererConfig
from signerf.diffuser.diffuser import Diffuser, DiffuserConfig
from signerf.utils.image_tensor_converter import tensor_to_image, image_to_tensor
from signerf.utils.intersection import intersect_with_aabb

CONSOLE = Console(width=120)

@dataclass
class DatasetGeneratorConfig(cfg.InstantiateConfig):
    """Configuration for diffuser instantiation"""

    _target: Type = field(default_factory=lambda: DatasetGenerator)
    """target class to instantiate"""
    path: Path = field(default_factory=lambda: Path("./generation"))
    """ Path to the generadet dataset parent directory"""
    dataset_name: str = field(default_factory=lambda: "experiment-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    """ Name of the generated dataset"""
    downscale_factor: int = field(default_factory=lambda: 2)
    """ Downscale factor for the generated dataset"""
    fx: Optional[float] = None
    """ Focal length in x direction"""
    fy: Optional[float] = None
    """ Focal length in y direction"""
    cx: Optional[float] = None
    """ Principal point in x direction"""
    cy: Optional[float] = None
    """ Principal point in y direction"""
    width: Optional[int] = None
    """ Image width"""
    height: Optional[int] = None
    """ Image height"""
    masking_mode: Literal["shape", "aabb"] = "aabb"
    """ Masking mode for the generated dataset"""
    aabb_min :List[float] = field(default_factory=lambda: [-0.1, -0.1, -0.1])
    """ Axis aligned bounding box for the aabb masking mode min values"""
    aabb_max :Tuple[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])
    """ Axis aligned bounding box for the aabb masking mode max values"""
    rows: int = 2
    """ Number of rows for the generated dataset reference sheet"""
    cols: int = 3
    """ Number of columns for the generated dataset reference sheet"""
    mask_dialation: Optional[Tuple[int, int]] = (50, 50) # Before (20, 20)
    """ Mask dialation for the shape masking mode"""
    additional_depth_radius: float = 0.1
    """ Additional depth radius for the shape masking mode"""
    renderer: RendererConfig = field(default_factory=lambda: RendererConfig)
    """ Render config for the generated dataset"""
    diffuser: DiffuserConfig =  field(default_factory=lambda: DiffuserConfig)
    """ Diffuser config for the generated dataset"""
    border_width_between_images: int = 0
    """ Border width between images in the generated dataset reference sheet"""
    inverse_mask: bool = False
    """ Inverse mask for the generated dataset"""
    manual_depth:  Optional[Tuple[int, int]] = None
    """ Manual depth for the generated dataset"""
    combine_shape_with_depth: bool = False
    """ Combine shape with NeRF depth for the generated dataset"""


class DatasetGenerator:
    """Diffuser class to diffuse images"""

    def __init__(
        self,
        config: DatasetGeneratorConfig,
        original_transform_matrix: Tensor,
        original_scale_factor: float,
        transform_poses_to_original_space: Callable[[Tensor], Tensor],
        device: str
    ) -> None:
        self.config = config
        self.device = device
        self.original_transform_matrix = original_transform_matrix
        self.original_scale_factor = original_scale_factor
        self.transform_poses_to_original_space = transform_poses_to_original_space

        self.path = config.path
        self.dataset_name = config.dataset_name

        self.fx = config.fx
        self.fy = config.fy
        self.cx = config.cx
        self.cy = config.cy
        self.width = config.width
        self.height = config.height
        self.downscale_factor = config.downscale_factor

        self.masking_mode = config.masking_mode
        self.aabb = torch.tensor([config.aabb_min, config.aabb_max], dtype=torch.float32, device=self.device)
        self.inverse_mask = config.inverse_mask
        self.combine_shape_with_depth = config.combine_shape_with_depth

        self.rows = config.rows
        self.cols = config.cols
        self.border_width_between_images = config.border_width_between_images

        self.mask_dialation = config.mask_dialation
        self.additional_depth_radius = config.additional_depth_radius
        self.manual_depth = config.manual_depth

        # Create diffuser and renderer
        self.renderer = Renderer(config.renderer, device=self.device)
        self.diffuser = Diffuser(config.diffuser, device=self.device)

        self.is_synthetic = False

        self.dataset_path = None
        self.images_path = None
        self.masks_path = None
        self.conditions_path = None
        self.rendered_path = None
        self.originals_path = None
        self.images_scaled_path = None
        self.masks_scaled_path = None
        self.conditions_scaled_path = None
        self.rendered_path_scaled = None
        self.originals_scaled_path = None
        self.references_path = None
        self.transforms_path = None


    def init_directory(self) -> None:
        """  Initialize the directory for the generated dataset """

        self.dataset_path = self.config.path / self.dataset_name
        self.dataset_path.mkdir(parents=True, exist_ok=True)

        self.images_path = self.dataset_path / "images"
        self.images_path.mkdir(parents=True, exist_ok=True)
        self.masks_path = self.dataset_path / "masks"
        self.masks_path.mkdir(parents=True, exist_ok=True)
        self.conditions_path = self.dataset_path / "conditions"
        self.conditions_path.mkdir(parents=True, exist_ok=True)
        self.rendered_path = self.dataset_path / "rendered"
        self.rendered_path.mkdir(parents=True, exist_ok=True)
        self.originals_path = self.dataset_path / "originals"
        self.originals_path.mkdir(parents=True, exist_ok=True)

        self.images_scaled_path = self.dataset_path / f"images_{self.downscale_factor}"
        self.images_scaled_path.mkdir(parents=True, exist_ok=True)
        self.masks_scaled_path = self.dataset_path / f"masks_{self.downscale_factor}"
        self.masks_scaled_path.mkdir(parents=True, exist_ok=True)
        self.conditions_scaled_path = self.dataset_path / f"conditions_{self.downscale_factor}"
        self.conditions_scaled_path.mkdir(parents=True, exist_ok=True)
        self.rendered_path_scaled = self.dataset_path / f"rendered_{self.downscale_factor}"
        self.rendered_path_scaled.mkdir(parents=True, exist_ok=True)
        self.originals_scaled_path = self.dataset_path / f"originals_{self.downscale_factor}"
        self.originals_scaled_path.mkdir(parents=True, exist_ok=True)

        self.references_path = self.dataset_path / "references"
        self.references_path.mkdir(parents=True, exist_ok=True)

        self.transforms_path = self.dataset_path / "transforms.json"

        # Save config in dataset directory
        config_yaml_path = self.dataset_path / "config.yml"
        CONSOLE.log(f"Saving config to: {config_yaml_path}")
        config_yaml_path.write_text(yaml.dump(self.config), "utf8")


    def generate_dataset(
        self,
        graph: Model,
        reference_camera_to_worlds: torch.Tensor["batch", 3, 4],
        original_dataset: Optional[InputDataset] = None,
        synthetic_camera_to_worlds: Optional[torch.Tensor["batch", 3, 4]] = None,
        merge_with_original_dataset: bool = False,
    ) -> None:
        """
            Generate a dataset with the given parameters

            Args:
                graph: The NeRF model to use for rendering
                original_dataset: The original dataset to either generate or merge with
                camera_to_worlds: The camera to world matrices for additional cameras
                reference_camera_to_worlds: The camera to world matrices for the reference cameras
                synthetic_camera_to_worlds: Whether to add the original dataset with inverse masks to the generated dataset
            Returns:
                None


        """

        if original_dataset is None and synthetic_camera_to_worlds is None:
            raise ValueError("Either original dataset or camera_to_worlds must be given")

        if merge_with_original_dataset:
            if original_dataset is None or synthetic_camera_to_worlds is None:
                raise ValueError("Original dataset and camera_to_worlds must be given to merge with original dataset")
        else:
            if original_dataset is None and synthetic_camera_to_worlds is None:
                raise ValueError("Original dataset or camera_to_worlds must be given")


        # Log the config
        CONSOLE.print("[bold green]Dataset generation config[/bold green]")
        CONSOLE.print(self.config)
        CONSOLE.print()

        # Setup everything
        self.init_directory()
        self.renderer.setup()

        # Check if the dataset is synthetic
        if synthetic_camera_to_worlds is not None:
            self.is_synthetic = True

        # Dataset Size
        dataset_size = 0
        if merge_with_original_dataset:
            dataset_size = synthetic_camera_to_worlds.shape[0]
        elif original_dataset is not None:
            dataset_size = original_dataset.cameras.size
        else:
            dataset_size = synthetic_camera_to_worlds.shape[0]


        # Generating dataset into directory
        CONSOLE.print("[bold green]Generating dataset...[/bold green]")
        CONSOLE.print(f"Dataset name: {self.dataset_name}")
        CONSOLE.print(f"Dataset path: {self.dataset_path}")
        CONSOLE.print()
        CONSOLE.print(f"Reference image count: {self.rows * self.cols - 1}, rows: {self.rows}, cols: {self.cols}")
        CONSOLE.print(f"Generation image count: {dataset_size}")
        if merge_with_original_dataset:
            CONSOLE.print(f"Merging original dataset, image count: {original_dataset.cameras.size}")
        CONSOLE.print()
        if merge_with_original_dataset:
            CONSOLE.print(f"Total dataset size: {dataset_size + self.rows * self.cols - 1 + original_dataset.cameras.size}")
        else:
            CONSOLE.print(f"Total dataset size: {dataset_size + self.rows * self.cols - 1}")


        # Save time to evaluate the time it takes to generate the dataset
        start_time = time.time()

        # Calculate scaled image size
        scaled_image_width = int(self.width // self.downscale_factor)
        scaled_image_height = int(self.height // self.downscale_factor)


        # Reference cameras
        reference_cameras = Cameras(reference_camera_to_worlds, self.fx, self.fy, self.cx, self.cy, self.width, self.height)
        reference_cameras = reference_cameras.to(self.device)

        # Cameras
        cameras = None
        original_filenames: List[Path | None] | None = None

        # Original dataset
        if original_dataset is not None:
            cameras = original_dataset.cameras
            original_filenames = original_dataset._dataparser_outputs.image_filenames # pylint: disable=protected-access

        if synthetic_camera_to_worlds is not None:
            # Overwrite cameras as for merging we need the original cameras later
            cameras = Cameras(synthetic_camera_to_worlds, self.fx, self.fy, self.cx, self.cy, self.width, self.height)
            original_filenames = [None] * synthetic_camera_to_worlds.shape[0]
        cameras = cameras.to(self.device)

        # Transforms
        transforms = {
            "camera_model": "OPENCV",
            "orientation_override": "none",
            "method": "SIGNeRF",
            "is_synthetic": self.is_synthetic,
            "is_combined": merge_with_original_dataset,
            "frames": [],
            "original_transform_matrix": self.original_transform_matrix.cpu().numpy().tolist(),
            "original_scale_factor": self.original_scale_factor
        }

        # Generate reference sheet
        image_reference_sheet, mask_reference_sheet, condition_reference_sheet, edited_reference_sheet, references = self.generate_reference_sheet(graph, reference_cameras, scaled_image_width, scaled_image_height)

        # Save reference sheet
        tensor_to_image(image_reference_sheet).save(self.references_path / "image_reference_sheet.png")
        tensor_to_image(mask_reference_sheet).save(self.references_path / "mask_reference_sheet.png")
        tensor_to_image(condition_reference_sheet).save(self.references_path / "condition_reference_sheet.png")
        tensor_to_image(edited_reference_sheet).save(self.references_path / "edited_reference_sheet.png")

        # Eidted image idx
        edited_image_idx = 0

        with Progress( TextColumn("[progress.description]{task.description}"), BarColumn(), TimeElapsedColumn(), MofNCompleteColumn(), transient=True) as progress:
            task = progress.add_task("[green] Saving reference images...", total=len(reference_cameras))
            transforms["reference_indices"] = []

            # Save reference images
            for i, camera in enumerate(reference_cameras):
                transforms = self.save_generated_images(edited_image_idx, references[i], camera, transforms)
                transforms["reference_indices"].append(edited_image_idx)

                edited_image_idx += 1

                progress.advance(task)

            # Save transforms
            with open(self.transforms_path, "w") as file: # pylint: disable=unspecified-encoding
                json.dump(transforms, file, indent=4)

        with Progress( TextColumn("[progress.description]{task.description}"), BarColumn(), TimeElapsedColumn(), MofNCompleteColumn(), transient=True) as progress:
            task = progress.add_task("[green] Generating and saving remaining images...", total=len(cameras))
            transforms["generated_indices"] = []

            # Generate dataset
            for i, camera in enumerate(cameras):
                filename = original_filenames[i]
                images = self.generate_with_reference_sheet(graph, camera, filename, scaled_image_width, scaled_image_height, edited_reference_sheet, condition_reference_sheet)
                transforms = self.save_generated_images(edited_image_idx, images, camera, transforms, filename is not None)
                transforms["generated_indices"].append(edited_image_idx)

                edited_image_idx += 1
                progress.advance(task)

            # Save transforms
            with open(self.transforms_path, "w") as file: # pylint: disable=unspecified-encoding
                json.dump(transforms, file, indent=4)

        if merge_with_original_dataset:
            with Progress( TextColumn("[progress.description]{task.description}"), BarColumn(), TimeElapsedColumn(), MofNCompleteColumn(), transient=True) as progress:
                task = progress.add_task("[green] Merging dataset into generated dataset ...", total=original_dataset.cameras.size)
                transforms["original_indices"] = []

                # Merge original dataset into generated dataset
                for idx in range(original_dataset.cameras.size): # pylint: disable=consider-using-enumerate

                    # Get image and camera from dataset
                    image = original_dataset.get_image_float32(idx)
                    cameras = original_dataset.cameras
                    camera = cameras[idx]
                    camera = camera.to(graph.device)

                    # Render image with NeRF and create mask
                    render, mask, condition = self.render_camera(graph, camera, combine_shape_with_depth=self.combine_shape_with_depth) # pylint: disable=unbalanced-tuple-unpacking

                    # Invert mask, since they don't conain the object
                    mask = ~mask # pylint: disable=invalid-unary-operand-type

                    # Scale images
                    image_scaled = F.interpolate(image.permute(2,0,1).unsqueeze(0), (scaled_image_height, scaled_image_width), mode="bilinear", align_corners=False).squeeze(0).permute(1,2,0)
                    render_scaled = F.interpolate(render.permute(2,0,1).unsqueeze(0), (scaled_image_height, scaled_image_width), mode="bilinear", align_corners=False).squeeze(0).permute(1,2,0)
                    mask_scaled = F.interpolate(mask.float().permute(2,0,1).unsqueeze(0), (scaled_image_height, scaled_image_width), mode="bilinear", align_corners=False).squeeze(0).permute(1,2,0) > 0.5
                    condition_scaled = F.interpolate(condition.permute(2,0,1).unsqueeze(0), (scaled_image_height, scaled_image_width), mode="bilinear", align_corners=False).squeeze(0).permute(1,2,0)

                    images = {
                        "render": render,
                        "mask": mask,
                        "condition": condition,
                        "edited": image,
                        "render_scaled": render_scaled,
                        "mask_scaled": mask_scaled,
                        "condition_scaled": condition_scaled,
                        "edited_scaled": image_scaled,
                    }

                    transforms = self.save_generated_images(edited_image_idx, images, camera, transforms, True)
                    transforms["original_indices"].append(edited_image_idx)

                    edited_image_idx += 1
                    progress.advance(task)

            # Save transforms
            with open(self.transforms_path, "w") as file: # pylint: disable=unspecified-encoding
                json.dump(transforms, file, indent=4)

        # Log time
        end_time = time.time()
        CONSOLE.print("[bold green]Successfully generated dataset in {:.2f} minutes".format((end_time - start_time) / 60))




    def save_generated_images(self, idx: int, images: Dict[str, Tensor], camera: Cameras, current_transforms: Dict[str, Any], is_original: bool = False) -> Dict[str, Any]:
        """
        Save the generated images

        Args:
            images (Dict[str, Tensor]): Dictionary containing the generated images
            idx (int): Index of the image
            camera (Camera): Camera used to generate the images
            current_transforms (Dict[str, Any]): Current transforms
            is_original (bool, optional): Whether the image is an original image. Defaults to False.

        Returns:
            Dict[str, Any]: Dictionary containing the transforms
        """

        # Save images

        # Check if images has key "edited"
        if "edited" in images:
            tensor_to_image(images["edited"]).save(self.images_path / f"image_{idx}.png")

        if "render" in images:
            if is_original:
                tensor_to_image(images["render"]).save(self.originals_path / f"image_{idx}.png")
            else:
                tensor_to_image(images["render"]).save(self.rendered_path / f"image_{idx}.png")

        if "mask" in images:
            tensor_to_image(images["mask"]).save(self.masks_path / f"mask_{idx}.png")

        if "condition" in images:
            tensor_to_image(images["condition"]).save(self.conditions_path / f"condition_{idx}.png")


        # Save scaled images
        if "edited_scaled" in images:
            tensor_to_image(images["edited_scaled"]).save(self.images_scaled_path / f"image_{idx}.png")

        if "render_scaled" in images:
            if is_original:
                tensor_to_image(images["render_scaled"]).save(self.rendered_path_scaled / f"image_{idx}.png")
            else:
                tensor_to_image(images["render_scaled"]).save(self.rendered_path_scaled / f"image_{idx}.png")

        if "mask_scaled" in images:
            tensor_to_image(images["mask_scaled"]).save(self.masks_scaled_path / f"mask_{idx}.png")

        if "condition_scaled" in images:
            tensor_to_image(images["condition_scaled"]).save(self.conditions_scaled_path / f"condition_{idx}.png")

        # Transforms in original splace
        transform_matrix = self.transform_poses_to_original_space(camera.camera_to_worlds.cpu().unsqueeze(0)).squeeze(0)
        transform_matrix = torch.cat([transform_matrix.cpu(), torch.tensor([[0.0, 0.0, 0.0, 1.0]])], dim=0)

        scene_transform_matrix = torch.cat([camera.camera_to_worlds.cpu(), torch.tensor([[0.0, 0.0, 0.0, 1.0]])], dim=0)

        # Append transforms
        current_transforms["frames"].append({
            "fl_x": camera.fx.item(),
            "fl_y": camera.fy.item(),
            "cx": camera.cx.item(),
            "cy": camera.cy.item(),
            "w": camera.width.item(),
            "h": camera.height.item(),
            "file_path": f"./images/image_{idx}.png",
            "_mask_path": f"./masks/mask_{idx}.png", # Lover underscore to avoid confusion with mask in nerfacto dataparser
            "transform_matrix": scene_transform_matrix.cpu().numpy().tolist(), # FIXME: This should be transforms in original space but need to fix
            "scene_transform_matrix": scene_transform_matrix.cpu().numpy().tolist(),
        })

        return current_transforms

    def generate_reference_sheet(
        self,
        graph: Model,
        cameras: Cameras,
        scaled_image_width: int,
        scaled_image_height: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        """
        Generate a reference sheet with the given parameters

        Args:
            graph (Model): Graph to render the reference sheet with
            cameras (Cameras): Cameras to render the reference sheet with
            fielnames (List[str | None]): List of original reference filenames
            scaled_image_width (int): Scaled image width
            scaled_image_height (int): Scaled image height
            diffuser (Diffuser): Diffuser to use
            renderer (Renderer, optional): Renderer to use. Defaults to None.


        Returns:
            Tuple[Tensor,Tensor,Tensor]: Tuple containing the reference sheet, the reference sheet mask and the reference sheet conditions
        """
        # Check if camera count is equal to (rows * cols) - 1
        if len(cameras) != (self.rows * self.cols) - 1:
            raise ValueError(f"Camera count {len(cameras)} is not equal to (rows * cols) - 1 = {(self.rows * self.cols) - 1}")

        # Referene sheet width and height
        reference_sheet_width = int(self.cols * scaled_image_width) + int((self.cols - 1) * self.border_width_between_images)
        reference_sheet_height = int(self.rows * scaled_image_height) + int((self.rows - 1) * self.border_width_between_images)

        # Padd reference sheet width and height to be divisible by 8
        reference_sheet_width = int(math.ceil(reference_sheet_width / 8) * 8)
        reference_sheet_height = int(math.ceil(reference_sheet_height / 8) * 8)

        # Create empty reference sheets
        image_reference_sheet = torch.ones((reference_sheet_height, reference_sheet_width, 3), dtype=torch.float32, device=self.device)
        mask_reference_sheet = torch.zeros((reference_sheet_height, reference_sheet_width, 1), dtype=torch.float32, device=self.device)
        condition_reference_sheet = torch.zeros((reference_sheet_height, reference_sheet_width, 1), dtype=torch.float32, device=self.device)

        # Collect single reference sheet images
        references = []

        with Progress( TextColumn("[progress.description]{task.description}"), BarColumn(), TimeElapsedColumn(), MofNCompleteColumn(), transient=True) as progress:
            task = progress.add_task("[green] Composing reference sheet...", total=len(cameras))

            # Loop over reference cameras
            for i, camera in enumerate(cameras):

                render, mask, condition = self.render_camera(graph, camera, combine_shape_with_depth=self.combine_shape_with_depth) # pylint: disable=unbalanced-tuple-unpacking

                # Calculate row and column index
                row = i // self.cols
                col = i % self.cols

                # Downscale image, mask and condition
                render_scaled = F.interpolate(render.permute(2,0,1).unsqueeze(0),(scaled_image_height, scaled_image_width), mode="bilinear", align_corners=False).squeeze(0).permute(1,2,0)
                mask_scaled = F.interpolate(mask.float().permute(2,0,1).unsqueeze(0), (scaled_image_height, scaled_image_width), mode="bilinear", align_corners=False).squeeze(0).permute(1,2,0) > 0.5
                condition_scaled = F.interpolate(condition.permute(2,0,1).unsqueeze(0), (scaled_image_height, scaled_image_width), mode="bilinear", align_corners=False).squeeze(0).permute(1,2,0)

                # Calculate image, mask and condition start and end indices
                image_start_row = row * scaled_image_height + row * self.border_width_between_images
                image_end_row = image_start_row + scaled_image_height
                image_start_col = col * scaled_image_width + col * self.border_width_between_images
                image_end_col = image_start_col + scaled_image_width

                # Add image, mask and condition to reference sheet
                image_reference_sheet[image_start_row:image_end_row, image_start_col:image_end_col, :] = render_scaled
                mask_reference_sheet[image_start_row:image_end_row, image_start_col:image_end_col, :] = mask_scaled
                condition_reference_sheet[image_start_row:image_end_row, image_start_col:image_end_col, :] = condition_scaled

                # Store reference
                references.append({
                    "render": render,
                    "mask": mask,
                    "condition": condition,
                    "render_scaled": render_scaled,
                    "mask_scaled": mask_scaled,
                    "condition_scaled": condition_scaled,
                })

                progress.advance(task)


        with Progress( TextColumn("[progress.description]{task.description}"), BarColumn(), TimeElapsedColumn(), MofNCompleteColumn(), transient=True) as progress:
            task = progress.add_task("[green] Generating reference sheet ...", total=1)

            # Run diffuser on reference sheet
            edited_reference_sheet = self.diffuser.diffuse(image_reference_sheet, image_reference_sheet, mask_reference_sheet, condition_reference_sheet)

            # Combine edited and render
            edited_reference_sheet = edited_reference_sheet.to(self.device)
            edited_reference_sheet = edited_reference_sheet * mask_reference_sheet.repeat(1,1,3) + image_reference_sheet * (1 - mask_reference_sheet.repeat(1,1,3))

            progress.advance(task)

        with Progress( TextColumn("[progress.description]{task.description}"), BarColumn(), TimeElapsedColumn(), MofNCompleteColumn(), transient=True) as progress:
            task = progress.add_task("[green] Adding edited images to reference sheet ...", total=len(cameras))

            # Isolate all reference sheet images
            for i, camera in enumerate(cameras):
                # Calculate row and column index
                row = i // self.cols
                col = i % self.cols

                # Calculate image, mask and condition start and end indices
                image_start_row = row * scaled_image_height + row * self.border_width_between_images
                image_end_row = image_start_row + scaled_image_height
                image_start_col = col * scaled_image_width + col * self.border_width_between_images
                image_end_col = image_start_col + scaled_image_width

                # Isolate image, mask and condition
                edited_scaled = edited_reference_sheet[image_start_row:image_end_row, image_start_col:image_end_col, :]

                # Upscale image, mask a
                edited = F.interpolate(edited_scaled.permute(2,0,1).unsqueeze(0), (self.height, self.width), mode="bilinear", align_corners=False).squeeze(0).permute(1,2,0)

                # Store reference
                references[i]["edited"] = edited
                references[i]["edited_scaled"] = edited_scaled

                progress.advance(task)

        return image_reference_sheet, mask_reference_sheet, condition_reference_sheet, edited_reference_sheet, references



    def generate_with_reference_sheet(
        self,
        graph: Model,
        camera: Cameras,
        filename: str | None,
        scaled_image_width: int,
        scaled_image_height: int,
        image_reference_sheet: Tensor["rows", "cols", 3],
        condition_reference_sheet: Tensor["rows", "cols", 1],
    ) -> Dict[str, Tensor]:
        """
        Generate a dataset with the given parameters

        Args:
            graph (Model): Graph to render the reference sheet with
            cameras (Cameras): Cameras to render the reference sheet with
            filenames (List[str | None]): Filenames of the reference images
            scaled_image_width (int): Width of the scaled reference images
            scaled_image_height (int): Height of the scaled reference images
            image_reference_sheet (Tensor["rows", "cols", 3]): Reference sheet to render the reference sheet with
            condition_reference_sheet (Tensor["rows", "cols", 1]): Condition reference sheet to render the reference sheet with

        Returns:
            Dict: Dictionary containing the generated images, masks, conditions and references
        """


        # Loop over reference cameras
        render, mask, condition = self.render_camera(graph, camera, combine_shape_with_depth=self.combine_shape_with_depth) # pylint: disable=unbalanced-tuple-unpacking

        # Load original reference image
        if filename is not None:
            render = Image.open(filename)
            render = image_to_tensor(render).to(self.device)

        # Downscale image, mask and condition
        render_scaled = F.interpolate(render.permute(2,0,1).unsqueeze(0), (scaled_image_height, scaled_image_width), mode="bilinear", align_corners=False).squeeze(0).permute(1,2,0)
        mask_scaled = F.interpolate(mask.float().permute(2,0,1).unsqueeze(0), (scaled_image_height, scaled_image_width), mode="bilinear", align_corners=False).squeeze(0).permute(1,2,0) > 0.5
        condition_scaled = F.interpolate(condition.permute(2,0,1).unsqueeze(0), (scaled_image_height, scaled_image_width), mode="bilinear", align_corners=False).squeeze(0).permute(1,2,0)

        # Add image, mask and condition to reference sheet at last position
        start_position_height = (self.rows - 1) * scaled_image_height + (self.rows - 1) * self.border_width_between_images
        end_position_height = start_position_height + scaled_image_height
        start_position_width = (self.cols - 1) * scaled_image_width + (self.cols - 1) * self.border_width_between_images
        end_position_width = start_position_width + scaled_image_width

        image_reference_sheet[start_position_height:end_position_height, start_position_width:end_position_width, :] = render_scaled
        mask_reference_sheet = torch.zeros_like(condition_reference_sheet)
        mask_reference_sheet[start_position_height:end_position_height, start_position_width:end_position_width, :] = mask_scaled
        condition_reference_sheet[start_position_height:end_position_height, start_position_width:end_position_width, :] = condition_scaled

        # Run diffuser on reference sheet
        edited_reference_sheet = self.diffuser.diffuse(image_reference_sheet, image_reference_sheet, mask_reference_sheet, condition_reference_sheet)

        # Isolate edited image
        edited_scaled = edited_reference_sheet[start_position_height:end_position_height, start_position_width:end_position_width, :]

        # Combine edited image with original image according to the mask
        edited_scaled = edited_scaled.to(self.device)
        edited_scaled = edited_scaled * mask_scaled + render_scaled * (~mask_scaled)

        # Upscale edited image
        edited = F.interpolate(edited_scaled.permute(2,0,1).unsqueeze(0), (self.height, self.width), mode="bilinear", align_corners=False).squeeze(0).permute(1,2,0)


        # Images dictionary
        images = {
            "render": render,
            "mask": mask,
            "condition": condition,
            "edited": edited,
            "render_scaled": render_scaled,
            "mask_scaled": mask_scaled,
            "condition_scaled": condition_scaled,
            "edited_scaled": edited_scaled,
        }

        return images


    def render_camera(self, graph: Model, camera: Cameras, with_mask: bool = True, with_condition: bool = True, combine_shape_with_depth: bool = False) -> Tuple[Tensor, Tensor | None, Tensor | None]:
        """
        Render a camera with the given parameters

        Args:
            graph (Model): Graph to render the reference sheet with
            camera (Cameras): Camera to render the reference sheet with
            with_mask (bool, optional): Whether to render the mask. Defaults to True.
            with_condition (bool, optional): Whether to render the condition. Defaults to True.

        Returns:
            Tuple[Tensor,Tensor,Tensor]: Tuple containing the reference sheet, the reference sheet mask and the reference sheet conditions
        """

        camera_ray_bundle = camera.generate_rays(camera_indices=0, aabb_box=graph.render_aabb)

        graph.eval()
        outputs = graph.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        graph.train()

        if outputs is None:
            raise RuntimeError("Render thread did not return any outputs")

        rgb_tensor = outputs["rgb"]
        depth_tensor = outputs["depth"]

        rgb_image = rgb_tensor
        mask_image = None
        condition_image = None

        if not with_mask:
            return rgb_tensor, None, None, None


        if self.masking_mode == "shape":
            # Throw if renderer is not set
            if self.renderer is None:
                raise ValueError("Renderer is None but masking mode is shape")

            _, depth = self.renderer.render_camera(camera)

            # Mask out pixels that are not in front of the object
            non_empty_space = depth > 0
            visible_mask = (depth < depth_tensor) * non_empty_space
            visible_mask = ~visible_mask if self.inverse_mask else visible_mask
            is_visible = torch.sum(visible_mask) > 1e-6

            if is_visible:

                # Dialate mask
                if(self.mask_dialation is not None):
                    visible_mask_np = visible_mask.cpu().numpy().astype(float)
                    visible_mask_np = cv2.dilate(visible_mask_np, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.mask_dialation))
                    mask_image = torch.tensor(visible_mask_np, dtype=torch.float32, device=graph.device).unsqueeze(-1) > 0
                else:
                    mask_image = visible_mask

                if not with_condition:
                    return rgb_tensor, mask_image, None, None

                # Composition depth
                if self.manual_depth is not None:
                    min_manual_depth = self.manual_depth[0]
                    max_manual_depth = self.manual_depth[1]
                else:
                    min_manual_depth = torch.min(depth[visible_mask * depth > 0])  - self.additional_depth_radius
                    max_manual_depth = torch.max(depth) + self.additional_depth_radius

                object_depth_normalized = (depth - min_manual_depth) / (max_manual_depth - min_manual_depth)
                nerf_depth_normalized = (depth_tensor - min_manual_depth) / (max_manual_depth - min_manual_depth)
                condition_image =  visible_mask * object_depth_normalized + (~visible_mask) * nerf_depth_normalized
                condition_image = 1 - torch.clamp(condition_image, 0, 1)

            else:
                mask_image = torch.zeros(self.height, self.width, 1, dtype=torch.bool, device=self.device) # Changed

                if not with_condition:
                    return rgb_tensor, mask_image, None, None

                condition_image = torch.zeros(self.height, self.width, 1, dtype=torch.float32, device=self.device) # Changed

        elif self.masking_mode == "aabb":
            rays_o = camera_ray_bundle.origins
            rays_d = camera_ray_bundle.directions

            # Get box
            nears, fars = intersect_with_aabb(rays_o, rays_d, self.aabb)

            # Get pure mask
            non_empty_space = (nears < fars) & (nears > 0.0) # FIXME: We ignore cameras inside the box
            visible_mask = (nears < depth_tensor) * (depth_tensor < fars) * non_empty_space
            visible_mask = ~visible_mask if self.inverse_mask else visible_mask

            is_visible = torch.sum(visible_mask) > 1e-6

            if is_visible:

                # Dialate mask
                if self.mask_dialation is not None:
                    visible_mask_np = visible_mask.cpu().numpy().astype(float)
                    visible_mask_np = cv2.dilate(visible_mask_np, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.mask_dialation))
                    mask_image = torch.tensor(visible_mask_np, dtype=torch.float32, device=graph.device).unsqueeze(-1) > 0
                else:
                    mask_image = visible_mask

                if not with_condition:
                    return rgb_tensor, mask_image, None, None

                # Composition depth
                if self.manual_depth is not None:
                    min_manual_depth = self.manual_depth[0]
                    max_manual_depth = self.manual_depth[1]
                else:
                    masked_non_zero_depth= depth_tensor[(depth_tensor * visible_mask) > 0]
                    min_manual_depth = torch.min(masked_non_zero_depth[masked_non_zero_depth > 0]) - self.additional_depth_radius
                    max_manual_depth = torch.max(masked_non_zero_depth) + self.additional_depth_radius

                if combine_shape_with_depth:
                    if self.renderer is None:
                        raise ValueError("Renderer is None but masking mode is shape")

                    color, depth = self.renderer.render_camera(camera)
                    non_empty_space_nerf = depth > 0
                    camera_visible_mask = (depth < depth_tensor) * non_empty_space_nerf

                    object_depth_normalized = (depth - min_manual_depth) / (max_manual_depth - min_manual_depth)
                    nerf_depth_normalized = (depth_tensor - min_manual_depth) / (max_manual_depth - min_manual_depth)

                    isolated_color_channel = color[:,:,0].reshape(color.shape[0], color.shape[1], 1) / 255.0
                    condition_image =  camera_visible_mask * isolated_color_channel + (~camera_visible_mask) * nerf_depth_normalized
                    condition_image = 1 - torch.clamp(condition_image, 0, 1)
                else:
                    depth_normalized = (depth_tensor - min_manual_depth) / (max_manual_depth - min_manual_depth)
                    condition_image = 1 - torch.clamp(depth_normalized, 0, 1)

            else:
                mask_image = torch.zeros(self.height, self.width, 1, dtype=torch.bool, device=self.device) # Changed

                if not with_condition:
                    return rgb_tensor, mask_image, None, None

                condition_image = torch.zeros(self.height, self.width, 1, dtype=torch.float32, device=self.device) # Changed

        return rgb_image, mask_image, condition_image
