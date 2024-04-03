""" Patch-based pixel sampler for SIGNeRF dataset. """

from dataclasses import dataclass, field
from typing import Optional, Type, Union

import torch
from jaxtyping import Int
from torch import Tensor


from nerfstudio.data.pixel_samplers import PixelSampler, PixelSamplerConfig


@dataclass
class PatchPixelSamplerConfig(PixelSamplerConfig):
    """Config dataclass for PatchPixelSampler."""

    _target: Type = field(default_factory=lambda: PatchPixelSampler)
    """Target class to instantiate."""
    patch_size: int = 32
    """Side length of patch. This must be consistent in the method
    config in order for samples to be reshaped into patches correctly."""


class PatchPixelSampler(PixelSampler):
    """Samples 'pixel_batch's from 'image_batch's. Samples square patches
    from the images randomly. Useful for patch-based losses.

    Args:
        config: the PatchPixelSamplerConfig used to instantiate class
    """

    config: PatchPixelSamplerConfig

    def set_num_rays_per_batch(self, num_rays_per_batch: int):
        """Set the number of rays to sample per batch. Overridden to deal with patch-based sampling.

        Args:
            num_rays_per_batch: number of rays to sample per batch
        """
        self.num_rays_per_batch = (num_rays_per_batch // (self.config.patch_size**2)) * (self.config.patch_size**2)

    # overrides base method
    def sample_method(
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        if isinstance(mask, Tensor) and not self.config.ignore_mask:
            # FIXME: Removed eroded mask sampling for now, as it is to slow
            # See: https://github.com/nerfstudio-project/nerfstudio/issues/3040
            # If there is a mask, sampling reduces back to standard sampling
            indices = super().sample_method(batch_size, num_images, image_height, image_width, mask=mask, device=device)
        else:
            sub_bs = batch_size // (self.config.patch_size**2)
            indices = torch.rand((sub_bs, 3), device=device) * torch.tensor(
                [num_images, image_height - self.config.patch_size, image_width - self.config.patch_size],
                device=device,
            )

            indices = (
                indices.view(sub_bs, 1, 1, 3)
                .broadcast_to(sub_bs, self.config.patch_size, self.config.patch_size, 3)
                .clone()
            )

            yys, xxs = torch.meshgrid(
                torch.arange(self.config.patch_size, device=device), torch.arange(self.config.patch_size, device=device)
            )
            indices[:, ..., 1] += yys
            indices[:, ..., 2] += xxs

            indices = torch.floor(indices).long()
            indices = indices.flatten(0, 2)

        return indices