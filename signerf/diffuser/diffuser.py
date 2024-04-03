""" Diffuser class to diffuse images """

from typing import Literal, Type
from dataclasses import dataclass, field

import json
import torch
import requests
from PIL import Image
from rich.progress import Console

from nerfstudio.configs import base_config as cfg

from signerf.utils.image_tensor_converter import tensor_to_image, image_to_tensor
from signerf.utils.image_base64_converter import image_to_base_64, base64_to_image

CONSOLE = Console(width=120)

@dataclass
class DiffuserConfig(cfg.InstantiateConfig):
    """Configuration for diffuser instantiation"""

    _target: Type = field(default_factory=lambda: Diffuser)
    """target class to instantiate"""
    mode : Literal["custom", "remoteSDWebUIControlNet"] = "remoteSDWebUIControlNet"
    """ wether to run locally or remotely """
    url : str = "http://127.0.0.1"
    """ url of the remote server """
    port : int = 5000
    """ port of the remote server """
    prompt: str = "don't change the image"
    """ instruction to use for the diffusion model """
    guidance_scale: float = 7
    """(text) guidance scale """
    image_guidance_scale: float = 1.5
    """image guidance scale """
    denoising_strength: float = 0.9
    """denoising strength for stable diffusion"""
    num_inference_steps: int = 20
    """Number of diffusion steps to take"""
    lower_bound: float = 0.02
    """Lower bound for diffusion timesteps to use for image editing"""
    upper_bound: float = 0.98
    """Upper bound for diffusion timesteps to use for image editing"""
    seed : int = 1
    """Seed for random number generator"""
    stable_diffusion_model : str = "sd_xl_base_1.0.safetensors [31e35c80fc]"
    """stable diffusion model to use"""
    controlnet_model : str = "diffusers_xl_depth_full [2f51180b]"
    """controlnet model to use"""
    controlnet_lowvram: bool = False
    """Whether to use low vram mode for controlnet"""
    controlnet_conditioning_scale: float = 0.8
    """controlnet condition scale for following the condition"""
    controlnet_conditioning_scale_start: float = 0.0
    """controlnet ratio of generation where this unit starts to have an effect"""
    controlnet_conditioning_scale_end: float = 1.0
    """controlnet ratio of generation where this unit ends to have an effect"""
    controlnet_control_mode: Literal["Balanced", "My prompt is more important", "ControlNet is more important"] = "Balanced"
    """controlnet control mode for balancing the controlnet and the prompt"""

class Diffuser:
    """Diffuser class to diffuse images"""

    def __init__(self, config: DiffuserConfig, device: str ) -> None:
        self.config = config
        self.device = device

        # General
        self.prompt = config.prompt
        self.guidance_scale = config.guidance_scale
        self.image_guidance_scale = config.image_guidance_scale
        self.denoising_strength = config.denoising_strength
        self.num_inference_steps = config.num_inference_steps
        self.seed = config.seed

        # Stable Diffusion
        self.stable_diffusion_model = config.stable_diffusion_model

        # ControlNet
        self.controlnet_conditioning_scale = config.controlnet_conditioning_scale
        self.controlnet_model = config.controlnet_model
        self.controlnet_lowvram = config.controlnet_lowvram
        self.controlnet_conditioning_scale_start = config.controlnet_conditioning_scale_start
        self.controlnet_conditioning_scale_end = config.controlnet_conditioning_scale_end
        self.controlnet_control_mode = config.controlnet_control_mode

        # Server
        self.url = f"{self.config.url}:{self.config.port}"


    def diffuse(
        self,
        original_image: torch.Tensor,
        rendered_image: torch.Tensor,
        mask_image: torch.Tensor = None,
        condition_image : torch.Tensor = None
    ) -> torch.Tensor:
        """Diffuse the image"""

        # Switch between diffuser
        if self.config.mode == "custom":
            self._custom(original_image, condition_image, rendered_image, mask_image)
            raise ValueError(f"Here you could implement your custom diffuser. But the mode {self.config.mode} is not supported.")
        elif self.config.mode == "remoteSDWebUIControlNet":
            return self._diffuse_remote_sdwebui_controlnet(original_image, condition_image, rendered_image, mask_image)


    def _custom(self, original_image, condition_image, rendered_image, mask_image):
        """Diffuse the image custom"""

        # TODO: Implement your custom diffuser here
        raise ValueError(f"Here you could implement your custom diffuser. Diffuser mode {self.config.mode} is not supported yet")


    def _diffuse_remote_sdwebui_controlnet(self, original_image, condition_image, rendered_image, mask_image) -> torch.Tensor:
        """Diffuse the image remotely with SDWebUI and ControlNet"""
        # ssh -fN -L 5000:localhost:5000 cgpool1913

        # Convert tensor to PIL image
        original_image_pil = tensor_to_image(original_image)
        rendered_image_pil = tensor_to_image(rendered_image)
        condition_image_pil = tensor_to_image(condition_image)
        mask_image_pil = tensor_to_image(mask_image) if mask_image is not None else None

        # Convert PIL image to base64
        original_image_bytes = image_to_base_64(original_image_pil)
        rendered_image_bytes = image_to_base_64(rendered_image_pil)
        condition_image_bytes = image_to_base_64(condition_image_pil)
        mask_image_bytes = image_to_base_64(mask_image_pil) if mask_image_pil is not None else None

        payload = {
            "init_images": [original_image_bytes],
            "model": self.stable_diffusion_model,
            "init_latent_images": [rendered_image_bytes],
            "prompt": self.prompt,
            "steps": self.num_inference_steps,
            "cfg_scale": self.guidance_scale,
            "image_cfg_scale": self.image_guidance_scale,
            "height": original_image.shape[0],
            "width": original_image.shape[1],
            "denoising_strength": self.denoising_strength,
            "seed": self.seed,
            "sampler_name": "Euler a",
            "alwayson_scripts": {
                "controlnet": {
                    "args": [
                        {
                            "input_image": condition_image_bytes,
                            "model": self.controlnet_model,
                            "module": "none",
                            "weight": self.controlnet_conditioning_scale,
                            "guidance_start": self.controlnet_conditioning_scale_start,
                            "guidance_end": self.controlnet_conditioning_scale_end,
                            "lowvram": self.controlnet_lowvram,
                            "control_mode": self.controlnet_control_mode
                        }
                    ]
                }
            }
        }

        if mask_image_bytes is not None:
            payload['mask'] = mask_image_bytes
            payload['mask_blur'] = 4
            payload['inpainting_fill'] = 1
            payload['inpaint_full_res'] = 0
            payload['inpaint_full_res_padding'] = 32


        header = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }

        assert self.url is not None, "URL is not set"

        try:
            req = requests.post(f'{self.url}/sdapi/v1/img2img', headers=header, data=json.dumps(payload), timeout=9999)
            res_json = req.json()
        except requests.exceptions.RequestException as e:
            CONSOLE.print(f"[bold red]Could not connect to server, is the server reachable at {self.url}:{self.config.port}?[/bold red]")
            print(e)
            return original_image

        # Status Code
        assert 'images' in res_json, f'Images not found in response: {res_json}'
        image = base64_to_image(res_json['images'][0])

        # Ensure size is correct by resizing to original image size (might be bad)
        image = image.resize((original_image.shape[1], original_image.shape[0]), Image.Resampling.LANCZOS)
        edited_image = image_to_tensor(image)

        return edited_image
