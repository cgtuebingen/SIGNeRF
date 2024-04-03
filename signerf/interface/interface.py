""" Interface for the SIGNeRF pipeline """

from collections import defaultdict
from typing import Callable, DefaultDict, List, Tuple, Optional
from pathlib import Path

import math
import torch
import numpy as np
import viser
import viser.transforms as vtf
from viser import MeshHandle
import trimesh
from rich.progress import Console

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.viewer.viewer_elements import (
    ViewerCheckbox,
    ViewerDropdown,
    ViewerElement,
    ViewerVec3,
    ViewerText,
)


from signerf.interface.viewer import Viewer
from signerf.signerf_pipeline import SIGNeRFPipeline
from signerf.utils.poses_generation import circle_poses, random_sphere_poses
from signerf.utils.image_tensor_converter import tensor_to_image
from signerf.utils.image_base64_converter import image_to_base_64
from signerf.interface.viewer_elements_extended import ViewerNumberStep as ViewerNumber

CONSOLE = Console(width=120)
VISER_NERFSTUDIO_SCALE_RATIO: float = 10.0

class Interface:
    """Interface for the SIGNeRF pipeline"""

    def __init__(
        self,
        viewer: Viewer,
        pipeline: SIGNeRFPipeline,
        rerender_cb: Callable[[], None],
        exchange_training_dataset_cb: Callable[[], None],
        trained = False,
        reference_camera_to_worlds: Optional[torch.Tensor] = None,
        synthetic_camera_to_worlds: Optional[torch.Tensor] = None,
        previous_experiment_given: bool = False,
    ):
        self.viewer = viewer
        self.viser_server = viewer.viser_server
        self._elements_by_tag: DefaultDict[str, List[ViewerElement]] = defaultdict(lambda: [])

        self.pipeline = pipeline
        self.dataset_generator = pipeline.dataset_generator

        self.reference_camera_count = self.dataset_generator.rows * self.dataset_generator.cols - 1

        if reference_camera_to_worlds is not None:
            self.reference_camera_to_worlds = reference_camera_to_worlds
        else:
            radius = 0.5
            self.reference_camera_to_worlds = circle_poses(
                size = self.reference_camera_count,
                radius= radius,
                device=torch.device("cpu"),
                theta=90.0,
                phi=(0, 300),
                position=[0, 0, 0],
                target=[0, 0, 0],
            )

        self.synthetic_cameras_handles = []

        if synthetic_camera_to_worlds is not None:
            self.synthetic_camera_to_worlds = synthetic_camera_to_worlds
            self.synthetic_camera_count = synthetic_camera_to_worlds.shape[0]
        else:
            self.synthetic_camera_to_worlds = None
            self.synthetic_camera_count = 40

        # VIEWER UI
        def handle_training_change(_):

            if self.dataset.value == "Synthetic Cameras" and self.synthetic_camera_to_worlds is None:
                CONSOLE.print("[bold red]Please generate the synthetic cameras first[/bold red]")
                return

            self.viewer.waiting_train.disabled = True

            original_dataset = self.pipeline.datamanager.train_dataset

            # Generate the dataset
            self.dataset_generator.generate_dataset(
                self.pipeline.model,
                reference_camera_to_worlds=self.reference_camera_to_worlds[:, :3, :],
                original_dataset= original_dataset if self.dataset.value == "Original Cameras" or self.combine_cameras.value else None,
                synthetic_camera_to_worlds=self.synthetic_camera_to_worlds[:, :3, :] if self.dataset.value == "Synthetic Cameras" else None,
                merge_with_original_dataset = self.combine_cameras.value if self.dataset.value == "Synthetic Cameras" else False,
            )

            # Change viewer button state
            self.viewer.waiting_train.visible = False
            self.viewer.pause_train.visible = True

            # Exchange the training dataset with the generated dataset
            exchange_training_dataset_cb()

            # Reload the model without proposal weights
            self.pipeline.load_model_with_proposal_weights = False
            self.pipeline.reload_model_state_dict_without_proposal_weights()

            # Change the training state
            self.viewer.trainer.training_state = "training"

        self.viewer.waiting_train.on_click(handle_training_change)

        # General UI
        def handle_directory_path_change(han):
            self.dataset_generator.path = Path(han.value)
            self.dataset_generator.config.path = Path(han.value)

        self.directory_path = ViewerText(
            "Directory Path",
            str(self.dataset_generator.path),
            cb_hook=handle_directory_path_change,
            hint="Path to the dataset",
            disabled=trained,
        )

        def handle_experiment_name_change(han):
            self.dataset_generator.dataset_name = han.value
            self.dataset_generator.config.dataset_name = han.value

        self.experiment_name = ViewerText(
            "Experiment Name",
            self.dataset_generator.dataset_name,
            cb_hook=handle_experiment_name_change,
            hint="Name of the experiment",
            disabled=trained,
        )

        # Reference Cameras UI
        self.reference_cameras_handles = []

        for idx in range(self.reference_camera_count):
            c2w = self.reference_camera_to_worlds.cpu().numpy()[idx]
            R = vtf.SO3.from_matrix(c2w[:3, :3])
            R = R @ vtf.SO3.from_x_radians(np.pi)
            camera_handle = self.viser_server.add_camera_frustum(
                name=f"/reference_cameras/camera_{idx:05d}",
                fov=float(2 * np.arctan(self.dataset_generator.cx / self.dataset_generator.fx)),
                scale=0.1,
                color=(255, 0, 0),
                aspect=float(self.dataset_generator.cx  / self.dataset_generator.cy),
                wxyz=R.wxyz,
                position=c2w[:3, 3] * VISER_NERFSTUDIO_SCALE_RATIO,
            )

            @camera_handle.on_click #pylint: disable=cell-var-from-loop
            def _(event: viser.SceneNodePointerEvent[viser.CameraFrustumHandle]) -> None:
                with event.client.atomic():
                    event.client.camera.position = event.target.position
                    event.client.camera.wxyz = event.target.wxyz

            self.reference_cameras_handles.append(camera_handle)

        self.reference_cameras_dropdown = ViewerDropdown(
            "Reference Cameras",
            "Camera 0",
            options=[f"Camera {idx}" for idx in range(self.reference_camera_count)],
            # cb_hook=lambda han: [rerender_cb()],
        )

        # Circle Setter UI
        if not trained:
            self.circle_setter_radius = ViewerNumber(
                "Radius",
                0.5,
                hint="Radius of the circle",
                step=0.01,
            )

            self.circle_setter_theta = ViewerNumber(
                "Theta",
                90.0,
                hint="Theta of the circle",
                step=0.01,
            )

            self.circle_setter_position = ViewerVec3(
                "Position",
                (0.0, 0.0, 0.0),
                hint="Position of the circle",
                step=0.01,
            )

            self.circle_setter_target = ViewerVec3(
                "Target",
                (0.0, 0.0, 0.0),
                hint="Target of the circle",
                step=0.01,
            )

        # Dataset Cameras UI
        self.dataset = ViewerDropdown(
            "Dataset",
            "Synthetic Cameras" if self.synthetic_camera_to_worlds is not None else "Original Cameras",
            ["Original Cameras", "Synthetic Cameras"],
            cb_hook=lambda han: [self.update_dataset_panel(han.value)],
            hint="Selection type",
            disabled=trained,
        )

        self.combine_cameras = ViewerCheckbox(
            "Combine w/ Original Cameras",
            True,
            hint="Use original cameras with cutout to keep the original scene alive during training",
            disabled=trained and self.dataset.value != "Original Cameras",
            visible=self.dataset.value == "Synthetic Cameras"
        )

        # Synthetic Cameras UI
        if (self.synthetic_camera_to_worlds is not None):
            self.synthetic_cameras_handles = []

            # FIXME: Maybe should load intrinsic parameters from the transforms
            for idx in range(self.synthetic_camera_count):
                c2w = self.synthetic_camera_to_worlds.cpu().numpy()[idx]
                R = vtf.SO3.from_matrix(c2w[:3, :3])
                R = R @ vtf.SO3.from_x_radians(np.pi)
                camera_handle = self.viser_server.add_camera_frustum(
                    name=f"/synthetic_cameras/camera_{idx:05d}",
                    fov=float(2 * np.arctan(self.dataset_generator.cx / self.dataset_generator.fx)),
                    scale=0.1,
                    color=(0, 255, 0),
                    aspect=float(self.dataset_generator.cx  / self.dataset_generator.cy),
                    wxyz=R.wxyz,
                    position=c2w[:3, 3] * VISER_NERFSTUDIO_SCALE_RATIO,
                )

                @camera_handle.on_click #pylint: disable=cell-var-from-loop
                def _(event: viser.SceneNodePointerEvent[viser.CameraFrustumHandle]) -> None:
                    with event.client.atomic():
                        event.client.camera.position = event.target.position
                        event.client.camera.wxyz = event.target.wxyz

                self.synthetic_cameras_handles.append(camera_handle)

        # Sphere Setter UI
        if not trained:
            self.sphere_camera_count = ViewerNumber(
                "Camera Count",
                self.synthetic_camera_count,
                hint="Number of cameras to generate",
                step=1,
                visible=(self.dataset.value == "Synthetic Cameras")
            )

            self.sphere_setter_radius = ViewerNumber(
                "Radius",
                0.5,
                hint="Radius of the sphere",
                step=0.01,
                visible=(self.dataset.value == "Synthetic Cameras")
            )

            self.sphere_setter_position = ViewerVec3(
                "Position",
                (0.0, 0.0, 0.0),
                hint="Position of the sphere",
                step=0.01,
                visible=(self.dataset.value == "Synthetic Cameras")
            )

            self.sphere_setter_target = ViewerVec3(
                "Target",
                (0.0, 0.0, 0.0),
                hint="Target of the sphere",
                step=0.01,
                visible=(self.dataset.value == "Synthetic Cameras")
            )


        # Selection Properties UI
        def handle_selection_change(han):
            self.update_selection_panel(han.value)
            self.dataset_generator.masking_mode = han.value.lower()
            self.dataset_generator.config.masking_mode = han.value.lower()

            self.update_selection_visibility(self.selection_visibility.value, han.value)


        self.selection = ViewerDropdown(
            "Selection",
            "AABB" if self.dataset_generator.masking_mode == "aabb" else "Shape",
            ["AABB", "Shape"],
            cb_hook=handle_selection_change,
            hint="Selection type",
            disabled=trained,
        )

        self.selection_visibility = ViewerCheckbox(
            "Visible",
            not trained,
            cb_hook=lambda han: [self.update_selection_visibility(han.value, self.selection.value)],
            hint="Visibility of the selection",
        )

        def handle_aabb_position_change(han):
            self.update_aabb(han.value, "position")
            aabb_min_max = self.calculate_aabb(self.aabb_position.value, self.aabb_size.value)
            self.dataset_generator.aabb = torch.tensor(aabb_min_max, dtype=torch.float32, device=self.dataset_generator.device)
            self.dataset_generator.config.aabb_min = aabb_min_max[0]
            self.dataset_generator.config.aabb_max = aabb_min_max[1]

        self.aabb_position = ViewerVec3(
            "Position",
            self.calculate_position_size(self.dataset_generator.aabb, "position"),
            cb_hook=handle_aabb_position_change,
            hint="Position of the AABB",
            visible=self.selection.value == "AABB",
            disabled=trained,
            step=0.01,
        )

        def handle_aabb_size_change(han):
            self.update_aabb(han.value, "size")
            aabb_min_max = self.calculate_aabb(self.aabb_position.value, self.aabb_size.value)
            self.dataset_generator.aabb = torch.tensor(aabb_min_max, dtype=torch.float32, device=self.dataset_generator.device).squeeze(0)
            self.dataset_generator.config.aabb_min = aabb_min_max[0]
            self.dataset_generator.config.aabb_max = aabb_min_max[1]

        self.aabb_size = ViewerVec3(
            "Size",
            self.calculate_position_size(self.dataset_generator.aabb, "size"),
            cb_hook=handle_aabb_size_change,
            hint="Size of the AABB",
            visible=self.selection.value == "AABB",
            disabled=trained,
            step=0.01,
        )

        def handle_shape_path_change(han):
            self.dataset_generator.renderer.object_path = han.value
            self.dataset_generator.renderer.config.object_path = han.value

        self.shape_path = ViewerText(
            "Path",
            self.dataset_generator.renderer.object_path,
            cb_hook=handle_shape_path_change,
            hint="Path to the shape",
            visible=self.selection.value == "Shape",
            disabled=trained,
        )

        def handle_shape_position_change(han):
            self.dataset_generator.renderer.position = [han.value[0], han.value[1], han.value[2]]
            self.dataset_generator.renderer.config.position = [han.value[0], han.value[1], han.value[2]]

            self.shape_transform.position = tuple(p * VISER_NERFSTUDIO_SCALE_RATIO for p in han.value)
            self.shape.position = tuple(p * VISER_NERFSTUDIO_SCALE_RATIO for p in han.value)

        shape_position = self.dataset_generator.renderer.position
        self.shape_position = ViewerVec3(
            "Position",
            (shape_position[0], shape_position[1], shape_position[2]),
            cb_hook=handle_shape_position_change,
            hint="Position of the shape",
            visible=self.selection.value == "Shape",
            disabled=trained,
            step=0.01,
        )

        def handle_shape_size_change(han):
            self.dataset_generator.renderer.scale = [han.value, han.value, han.value]
            self.dataset_generator.renderer.config.scale = [han.value, han.value, han.value]

            if self.shape is not None:
                self.shape.remove()

            obj_path = Path(self.shape_path.value)

            if obj_path.suffix != ".obj":
                CONSOLE.print(f"[bold red]Path {obj_path} is not an obj file[/bold red]")
                return

            if not obj_path.exists():
                CONSOLE.print(f"[bold red]Path {obj_path} does not exist[/bold red]")
                CONSOLE.print("Be sure that the path exists on the server not client")
                return

            vertices, faces = self.load_obj(obj_path, scale=self.shape_size.value * VISER_NERFSTUDIO_SCALE_RATIO)

            roll, pitch, yaw = self.shape_rotation.value

            # Degree to Radians
            roll = math.radians(roll)
            pitch = math.radians(pitch)
            yaw = math.radians(yaw)

            scaled_shape_position = tuple(p * VISER_NERFSTUDIO_SCALE_RATIO for p in self.shape_position.value)
            self.shape = self.viser_server.add_mesh(
                "shape",
                vertices,
                faces,
                position=scaled_shape_position,
                wxyz = vtf.SO3.from_rpy_radians(roll, pitch, yaw).wxyz,
                visible=(self.selection.value == "Shape" and not trained and self.selection_visibility.value)
            )


        shape_size = self.dataset_generator.renderer.scale
        self.shape_size = ViewerNumber(
            "Size",
            shape_size[0],
            cb_hook=handle_shape_size_change,
            hint="Size of the shape",
            visible=self.selection.value == "Shape",
            disabled=trained,
            step=0.01,
        )

        def handle_shape_rotation_change(han):
            self.dataset_generator.renderer.rotation = [han.value[0], han.value[1], han.value[2]]
            self.dataset_generator.renderer.config.rotation = [han.value[0], han.value[1], han.value[2]]

            # Degree to Radians
            radians = [math.radians(han.value[0]), math.radians(han.value[1]), math.radians(han.value[2])]

            self.shape_transform.wxyz = vtf.SO3.from_rpy_radians(*radians).wxyz
            self.shape.wxyz = vtf.SO3.from_rpy_radians(*radians).wxyz

        shape_rotation = self.dataset_generator.renderer.rotation
        self.shape_rotation = ViewerVec3(
            "Rotation",
            (shape_rotation[0], shape_rotation[1], shape_rotation[2]),
            cb_hook=handle_shape_rotation_change,
            hint="Rotation of the shape",
            visible=self.selection.value == "Shape",
            disabled=trained,
            step=0.01,
        )

        # Generation Properties UI
        def handle_prompt_change(han):
            self.dataset_generator.diffuser.config.prompt = han.value
            self.dataset_generator.diffuser.prompt = han.value

        self.prompt = ViewerText(
            "Prompt",
            self.dataset_generator.diffuser.config.prompt,
            cb_hook=handle_prompt_change,
            hint="Prompt for the generation",
            disabled=trained,
        )

        def handle_guidance_scale_change(han):
            self.dataset_generator.diffuser.config.guidance_scale = han.value
            self.dataset_generator.diffuser.guidance_scale = han.value

        self.guidance_scale = ViewerNumber(
            "Guidance Scale",
            self.dataset_generator.diffuser.config.guidance_scale,
            cb_hook=handle_guidance_scale_change,
            hint="Guidance scale for the generation",
            disabled=trained,
            step=0.1,
        )

        def handle_image_guidance_scale_change(han):
            self.dataset_generator.diffuser.config.image_guidance_scale = han.value
            self.dataset_generator.diffuser.image_guidance_scale = han.value

        self.image_guidance_scale = ViewerNumber(
            "Image Guidance Scale",
           self.dataset_generator.diffuser.config.image_guidance_scale,
            cb_hook=handle_image_guidance_scale_change,
            hint="Image guidance scale for the generation",
            disabled=trained,
            step=0.1,
        )

        def handle_denoising_strength_change(han):
            self.dataset_generator.diffuser.config.denoising_strength = han.value
            self.dataset_generator.diffuser.denoising_strength = han.value

        self.denoising_strength = ViewerNumber(
            "Denoising Strength",
            self.dataset_generator.diffuser.config.denoising_strength,
            cb_hook=handle_denoising_strength_change,
            hint="Denoising strength for the generation",
            disabled=trained,
            step=0.01,
        )

        def handle_num_inference_steps_change(han):
            self.dataset_generator.diffuser.config.num_inference_steps = han.value
            self.dataset_generator.diffuser.num_inference_steps = han.value

        self.num_inference_steps = ViewerNumber(
            "Inference Steps",
            self.dataset_generator.diffuser.config.num_inference_steps,
            cb_hook=handle_num_inference_steps_change,
            hint="Number of inference steps for the generation",
            disabled=trained,
            step=1.0,
        )

        def handle_seed_change(han):
            self.dataset_generator.diffuser.config.seed = han.value
            self.dataset_generator.diffuser.seed = han.value

        self.seed = ViewerNumber(
            "Seed",
            self.dataset_generator.diffuser.config.seed,
            cb_hook=handle_seed_change,
            hint="Seed for the generation",
            disabled=trained,
            step=1.0,
        )

        def handle_controlnet_conditioning_scale_change(han):
            self.dataset_generator.diffuser.config.controlnet_conditioning_scale = han.value
            self.dataset_generator.diffuser.controlnet_conditioning_scale = han.value

        self.controlnet_conditioning_scale = ViewerNumber(
            "ControlNet Conditioning Scale",
            self.dataset_generator.diffuser.config.controlnet_conditioning_scale,
            cb_hook=handle_controlnet_conditioning_scale_change,
            hint="ControlNet conditioning scale for the generation",
            disabled=trained,
            step=0.01,
        )


        # General Options UI
        with self.viser_server.add_gui_folder("General Options"):
            # self.viser_server.add_gui_markdown("<small> General properties </small> ")
            self.add_element(self.directory_path)
            self.add_element(self.experiment_name)

        # Reference Cameras UI
        with self.viser_server.add_gui_folder("Reference Cameras Options"):
            # self.viser_server.add_gui_markdown("<small> Reference cameras </small> ")
            self.add_element(self.reference_cameras_dropdown, additional_tags=("all",))
            self.reference_cameras_goto_camera = self.viser_server.add_gui_button(
                "Go to Camera",
                color='gray'
            )
            @self.reference_cameras_goto_camera.on_click
            def _goto_camera(event: viser.GuiEvent[viser.GuiButtonHandle]) -> None:
                idx = int(self.reference_cameras_dropdown.value.split(" ")[1])
                camera = self.reference_cameras_handles[idx]
                client_camera = self.viser_server.get_clients()[event.client_id].camera
                client_camera.position = camera.position
                client_camera.wxyz = camera.wxyz

            self.reference_cameras_set_to_camera = self.viser_server.add_gui_button(
                "Set to Camera",
                color='gray',
                disabled=trained,
            )

            @self.reference_cameras_set_to_camera.on_click
            def _set_to_camera(event: viser.GuiEvent[viser.GuiButtonHandle]) -> None:
                idx = int(self.reference_cameras_dropdown.value.split(" ")[1])
                camera = self.reference_cameras_handles[idx]
                client_camera = self.viser_server.get_clients()[event.client_id].camera
                camera.position = client_camera.position
                camera.wxyz = client_camera.wxyz

                R = vtf.SO3(wxyz=client_camera.wxyz)
                R = R @ vtf.SO3.from_x_radians(np.pi)
                R = torch.tensor(R.as_matrix())
                pos = torch.tensor(client_camera.position, dtype=torch.float64) / VISER_NERFSTUDIO_SCALE_RATIO
                c2w = torch.concatenate([R, pos[:, None]], dim=1)

                self.reference_camera_to_worlds[idx] = c2w

            if not trained:
                with self.viser_server.add_gui_folder("Circle Setter", expand_by_default=False):
                    self.add_element(self.circle_setter_radius)
                    self.add_element(self.circle_setter_theta)
                    self.circle_setter_phi = self.viser_server.add_gui_vector2(
                        "Phi",
                        (0.0, 300.0),
                        (-360.0, -360.0),
                        (360.0, 360.0),
                    )
                    self.add_element(self.circle_setter_position)
                    self.add_element(self.circle_setter_target)
                    self.circle_setter_generate = self.viser_server.add_gui_button(
                        "Generate Circle",
                        color='gray',
                    )

                    @self.circle_setter_generate.on_click
                    def _generate_circle(_) -> None:
                        self.reference_camera_to_worlds = circle_poses(
                            size = self.reference_camera_count,
                            radius=self.circle_setter_radius.value,
                            device=torch.device("cpu"),
                            theta=self.circle_setter_theta.value,
                            phi=self.circle_setter_phi.value,
                            position=self.circle_setter_position.value,
                            target=self.circle_setter_target.value,
                        )

                        for idx in range(self.reference_camera_count):
                            c2w = self.reference_camera_to_worlds.cpu().numpy()[idx]
                            R = vtf.SO3.from_matrix(c2w[:3, :3])
                            R = R @ vtf.SO3.from_x_radians(np.pi)
                            camera = self.reference_cameras_handles[idx]
                            camera.position = c2w[:3, 3] * VISER_NERFSTUDIO_SCALE_RATIO
                            camera.wxyz = R.wxyz


        # Dataset Cameras UI
        with self.viser_server.add_gui_folder("Dataset Cameras Options"):
            # self.viser_server.add_gui_markdown("<small> Dataset cameras </small> ")
            self.add_element(self.dataset, additional_tags=("all","dataset"))
            self.add_element(self.combine_cameras, additional_tags=("all","dataset"))

            if not trained:
                self.sphere_setter_folder = self.viser_server.add_gui_folder("Sphere Setter", expand_by_default=False)
                with self.sphere_setter_folder:
                    self.add_element(self.sphere_camera_count)
                    self.add_element(self.sphere_setter_radius)
                    self.sphere_setter_theta = self.viser_server.add_gui_vector2(
                        "Theta",
                        (0.0, 180.0),
                        (-180.0, -180.0),
                        (180.0, 180.0),
                        visible=(self.dataset.value == "Synthetic Cameras")
                    )
                    self.sphere_setter_phi = self.viser_server.add_gui_vector2(
                        "Phi",
                        (-180.0, 180.0),
                        (-360.0, -360.0),
                        (360.0, 360.0),
                        visible=(self.dataset.value == "Synthetic Cameras")
                    )
                    self.add_element(self.sphere_setter_position)
                    self.add_element(self.sphere_setter_target)
                    self.sphere_setter_generate = self.viser_server.add_gui_button(
                        "Generate Sphere",
                        color='gray',
                        visible=(self.dataset.value == "Synthetic Cameras")
                    )

                    @self.sphere_setter_generate.on_click
                    def _generate_sphere(_) -> None:
                        self.synthetic_camera_count = self.sphere_camera_count.value
                        self.synthetic_camera_to_worlds = random_sphere_poses(
                            size = self.synthetic_camera_count,
                            radius=self.sphere_setter_radius.value,
                            device=torch.device("cpu"),
                            theta=self.sphere_setter_theta.value,
                            phi=self.sphere_setter_phi.value,
                            position=self.sphere_setter_position.value,
                            target=self.sphere_setter_target.value,
                        )

                        for handle in self.synthetic_cameras_handles:
                            handle.remove()

                        for idx in range(self.synthetic_camera_count):
                            c2w = self.synthetic_camera_to_worlds.cpu().numpy()[idx]
                            R = vtf.SO3.from_matrix(c2w[:3, :3])
                            R = R @ vtf.SO3.from_x_radians(np.pi)
                            camera_handle = self.viser_server.add_camera_frustum(
                                name=f"/synthetic_cameras/camera_{idx:05d}",
                                fov=float(2 * np.arctan(self.dataset_generator.cx / self.dataset_generator.fx)),
                                scale=0.1,
                                color=(0, 255, 0),
                                aspect=float(self.dataset_generator.cx  / self.dataset_generator.cy),
                                wxyz=R.wxyz,
                                position=c2w[:3, 3] * VISER_NERFSTUDIO_SCALE_RATIO,
                                visible=(self.dataset.value == "Synthetic Cameras")
                            )

                            @camera_handle.on_click #pylint: disable=cell-var-from-loop
                            def _(event: viser.SceneNodePointerEvent[viser.CameraFrustumHandle]) -> None:
                                with event.client.atomic():
                                    event.client.camera.position = event.target.position
                                    event.client.camera.wxyz = event.target.wxyz

                            self.synthetic_cameras_handles.append(camera_handle)

        # Selection Options UI
        with self.viser_server.add_gui_folder("Selection Options"):
            # self.viser_server.add_gui_markdown("<small> Selection properties </small> ")
            self.add_element(self.selection, additional_tags=("all",))
            self.add_element(self.selection_visibility, additional_tags=("all",))


            # AABB
            self.add_element(self.aabb_position, additional_tags=("AABB",))
            self.add_element(self.aabb_size, additional_tags=("AABB",))
            scaled_aabb_position = tuple(p * VISER_NERFSTUDIO_SCALE_RATIO for p in self.aabb_position.value)
            scaled_aabb_size = tuple(p * VISER_NERFSTUDIO_SCALE_RATIO for p in self.aabb_size.value)
            self.aabb_transform = self.viser_server.add_transform_controls(
                "aabb_transform",
                depth_test=False,
                line_width=4.0,
                disable_rotations=True,
                position=scaled_aabb_position,
                visible=(self.selection.value == "AABB" and not trained and self.selection_visibility.value)
            )
            self.aabb = self.viser_server.add_box(
                "aabb_box",
                (0.0, 0.0, 0.0),
                position=scaled_aabb_position,
                dimensions=scaled_aabb_size,
                visible=(self.selection.value == "AABB" and not trained and self.selection_visibility.value)
            )

            @self.aabb_transform.on_update
            def _update_aabb_transform(_) -> None:
                pos = tuple(p / VISER_NERFSTUDIO_SCALE_RATIO for p in self.aabb_transform.position)
                self.aabb_position.value = (pos[0], pos[1], pos[2])
                self.aabb.position = self.aabb_transform.position

            # Shape
            self.add_element(self.shape_path, additional_tags=("Shape",))
            self.shape_load_button = self.viser_server.add_gui_button(
                "Load OBJ",
                color='gray'
            )

            self.shape: MeshHandle = None

            @self.shape_load_button.on_click
            def _shape_load(_) -> None:
                if self.shape is not None:
                    self.shape.remove()

                obj_path = Path(self.shape_path.value)

                if obj_path.suffix != ".obj":
                    CONSOLE.print(f"[bold red]Path {obj_path} is not an obj file[/bold red]")
                    return

                if not obj_path.exists():
                    CONSOLE.print(f"[bold red]Path {obj_path} does not exist[/bold red]")
                    CONSOLE.print("Be sure that the path exists on the server not client")
                    return

                vertices, faces = self.load_obj(obj_path, scale=self.shape_size.value * VISER_NERFSTUDIO_SCALE_RATIO)

                roll, pitch, yaw = self.shape_rotation.value

                # Degree to Radians
                roll = math.radians(roll)
                pitch = math.radians(pitch)
                yaw = math.radians(yaw)

                scaled_shape_position = tuple(p * VISER_NERFSTUDIO_SCALE_RATIO for p in self.shape_position.value)
                self.shape = self.viser_server.add_mesh(
                    "shape",
                    vertices,
                    faces,
                    position=scaled_shape_position,
                    wxyz = vtf.SO3.from_rpy_radians(roll, pitch, yaw).wxyz,
                    visible=(self.selection.value == "Shape" and not trained and self.selection_visibility.value)
                )

            # Load the shape on startup
            if (previous_experiment_given and self.selection.value == "Shape"):
                _shape_load(None)

            self.add_element(self.shape_size, additional_tags=("Shape",))
            self.add_element(self.shape_position, additional_tags=("Shape",))
            self.add_element(self.shape_rotation, additional_tags=("Shape",))

            scaled_shape_position = tuple(p * VISER_NERFSTUDIO_SCALE_RATIO for p in self.shape_position.value)
            self.shape_transform = self.viser_server.add_transform_controls(
                "shape_transform",
                depth_test=False,
                line_width=4.0,
                position=scaled_shape_position,
                wxyz=vtf.SO3.from_rpy_radians(*self.shape_rotation.value).wxyz,
                visible=(self.selection.value == "Shape" and not trained and self.selection_visibility.value)
            )

            @self.shape_transform.on_update
            def _update_shape_transform(_) -> None:
                pos = tuple(p / VISER_NERFSTUDIO_SCALE_RATIO for p in self.shape_transform.position)
                self.shape_position.value = (pos[0], pos[1], pos[2])
                roll, pitch, yaw = vtf.SO3(self.shape_transform.wxyz).as_rpy_radians()

                # Radians to Degree
                roll = math.degrees(roll)
                pitch = math.degrees(pitch)
                yaw = math.degrees(yaw)

                self.shape_rotation.value = (roll, pitch, yaw)

                if self.shape is not None:
                    self.shape.position = self.shape_transform.position
                    self.shape.wxyz = self.shape_transform.wxyz



        # Generation Options UI
        with self.viser_server.add_gui_folder("Generation Options"):
            # self.viser_server.add_gui_markdown("<small> Generation properties </small> ")
            self.add_element(self.prompt)
            self.add_element(self.guidance_scale)
            self.add_element(self.image_guidance_scale)
            self.add_element(self.denoising_strength)
            self.add_element(self.num_inference_steps)
            self.add_element(self.seed)
            self.add_element(self.controlnet_conditioning_scale)

        # Button to generate
        self.preview_generation_button = self.viser_server.add_gui_button(
            label="Preview Generation", disabled=trained, icon=viser.Icon.LAYOUT_COLLAGE, color='gray'
        )

        @self.preview_generation_button.on_click
        def _generate_preview(event: viser.GuiEvent[viser.GuiButtonHandle]) -> None:

            with event.client.add_gui_modal("Generating Reference Sheet") as modal:
                wait = event.client.add_gui_markdown("Please wait while the reference sheet is being generated...")

                reference_cameras = Cameras(
                    self.reference_camera_to_worlds[:, :3, :],
                    fx=self.dataset_generator.fx,
                    fy=self.dataset_generator.fy,
                    cx=self.dataset_generator.cx,
                    cy=self.dataset_generator.cy,
                    width=self.dataset_generator.width,
                    height=self.dataset_generator.height
                )

                reference_cameras = reference_cameras.to(self.dataset_generator.device)

                scaled_image_width = int(self.dataset_generator.width // self.dataset_generator.downscale_factor)
                scaled_image_height = int(self.dataset_generator.height // self.dataset_generator.downscale_factor)

                self.dataset_generator.renderer.setup()
                image_reference_sheet, mask_reference_sheet, condition_reference_sheet, edited_reference_sheet, references = self.dataset_generator.generate_reference_sheet(
                    self.pipeline.model,
                    reference_cameras,
                    scaled_image_width,
                    scaled_image_height
                )

                image = tensor_to_image(image_reference_sheet)
                mask = tensor_to_image(mask_reference_sheet)
                condition = tensor_to_image(condition_reference_sheet)
                edited = tensor_to_image(edited_reference_sheet)

                # Remove the wait message
                wait.remove()

                # Add the images to the modal
                image_base64 = image_to_base_64(image)
                mask_base64 = image_to_base_64(mask)
                condition_base64 = image_to_base_64(condition)
                edited_base64 = image_to_base_64(edited)

                image_html = f"<img src='data:image/png;base64,{image_base64}' width='100%' />"
                mask_html = f"<img src='data:image/png;base64,{mask_base64}' width='100%' />"
                condition_html = f"<img src='data:image/png;base64,{condition_base64}' width='100%' />"
                edited_html = f"<img src='data:image/png;base64,{edited_base64}' width='100%' />"


                for html in [(image_html, "Original Image"), (mask_html, "Mask Image"), (condition_html, "Condition Image"), (edited_html, "Edited Image")]:
                    event.client.add_gui_markdown(html[0])
                    event.client.add_gui_markdown(html[1])
                    event.client.add_gui_markdown(" ")

                exit_button = event.client.add_gui_button("Close")
                @exit_button.on_click
                def _(_) -> None:
                    modal.close()




    def add_element(self, e: ViewerElement, additional_tags: Tuple[str, ...] = tuple()) -> None:
        """Adds an element to the  panel

        Args:
            e: the element to add
            additional_tags: additional tags to add to the element for selection
        """
        self._elements_by_tag["all"].append(e)
        for t in additional_tags:
            self._elements_by_tag[t].append(e)
        e.install(self.viser_server)

    def update_selection_panel(self, selection: str) -> None:
        """Updates the selection panel based on the current selection

        Args:
            selection: the current selection
        """

        is_aabb_selected = selection == "AABB"
        is_shape_selected = selection == "Shape"

        self.aabb.visible = is_aabb_selected
        self.aabb_transform.visible = is_aabb_selected
        self.aabb_position.set_hidden(not is_aabb_selected)
        self.aabb_size.set_hidden(not is_aabb_selected)
        self.shape_position.set_hidden(not is_shape_selected)
        self.shape_size.set_hidden(not is_shape_selected)
        self.shape_rotation.set_hidden(not is_shape_selected)
        self.shape_path.set_hidden(not is_shape_selected)

    def update_dataset_panel(self, selection: str) -> None:
        """Updates the dataset panel based on the current selection

        Args:
            selection: the current selection
        """

        is_original_selected = selection == "Original Cameras"
        is_synthetic_selected = selection == "Synthetic Cameras"

        self.combine_cameras.set_hidden(is_original_selected)
        self.sphere_camera_count.set_hidden(is_original_selected)
        self.sphere_setter_radius.set_visible(is_synthetic_selected)
        self.sphere_setter_theta.visible = is_synthetic_selected
        self.sphere_setter_phi.visible = is_synthetic_selected
        self.sphere_setter_position.set_hidden(is_original_selected)
        self.sphere_setter_target.set_hidden(is_original_selected)
        self.sphere_setter_generate.visible = is_synthetic_selected

        for handle in self.synthetic_cameras_handles:
            handle.visible = is_synthetic_selected

    def update_selection_visibility(self, visibility: bool, selection_mode: str) -> None:
        """ Updates the visibility of the selection

        Args:
            visibility: the new visibility of the selection
            selection_mode: the selection mode
        """

        if selection_mode == "AABB":
            self.aabb.visible = visibility
            self.aabb_transform.visible = visibility

            if self.shape is not None:
                self.shape.visible = False
            self.shape_transform.visible = False

        elif selection_mode == "Shape":
            if self.shape is not None:
                self.shape.visible = visibility
            self.shape_transform.visible = visibility

            self.aabb.visible = False
            self.aabb_transform.visible = False


    def update_aabb(self, vec: Tuple[float, float, float], key: str) -> None:
        """Updates the AABB based on the current values of the AABB UI elements

        Args:
            han: the new value of the AABB UI element
            key: the key of the AABB UI element
        """
        if key == "position":
            self.aabb_transform.position = tuple(p * VISER_NERFSTUDIO_SCALE_RATIO for p in vec)
            self.aabb.position = tuple(p * VISER_NERFSTUDIO_SCALE_RATIO for p in vec)
        elif key == "size":
            self.aabb.remove()
            self.aabb = self.viser_server.add_box(
                "aabb_box",
                (0.0, 0.0, 0.0),
                position=tuple(p * VISER_NERFSTUDIO_SCALE_RATIO for p in self.aabb_position.value),
                dimensions=tuple(p * VISER_NERFSTUDIO_SCALE_RATIO for p in vec),
                visible=self.selection.value == "AABB"
            )

    def calculate_aabb(
        self,
        position: Tuple[float, float, float],
        size: Tuple[float, float, float]
    ) -> List[List[float]]:
        """Calculates the AABB based on the current values of the AABB UI elements

        Args:
            position: the position of the AABB
            size: the size of the AABB

        Returns:
            the calculated AABB
        """
        x_min = position[0] - size[0] / 2
        x_max = position[0] + size[0] / 2

        y_min = position[1] - size[1] / 2
        y_max = position[1] + size[1] / 2

        z_min = position[2] - size[2] / 2
        z_max = position[2] + size[2] / 2

        return [[x_min, y_min, z_min], [x_max, y_max, z_max]]

    def calculate_position_size(self, aabb: List[List[float]], key: str) -> Tuple[float, float, float]:
        """Calculates the position or scale based on the current values of the AABB

        Args:
            aabb: the AABB
            key: the key of the AABB UI element

        Returns:
            the calculated position or scale
        """
        if key == "position":
            return (aabb[0][0] + aabb[1][0]) / 2, (aabb[0][1] + aabb[1][1]) / 2, (aabb[0][2] + aabb[1][2]) / 2
        elif key == "size":
            return aabb[1][0] - aabb[0][0], aabb[1][1] - aabb[0][1], aabb[1][2] - aabb[0][2]


    def load_obj(self, path: str, scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Loads an obj file as a glb file

        Args:
            path: the path to the obj file on the server
            scale: the scale of the obj file

        Returns:
            the vertices and faces of the obj file

        """

        mesh = trimesh.load_mesh(path)
        assert isinstance(mesh, trimesh.Trimesh)

        mesh.apply_scale(scale * VISER_NERFSTUDIO_SCALE_RATIO)

        vertices = mesh.vertices
        faces = mesh.faces

        return vertices, faces