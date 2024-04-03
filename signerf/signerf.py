""" Model for SIGNeRF """

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type


import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from nerfstudio.model_components.losses import L1Loss, MSELoss, interlevel_loss
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig

@dataclass
class SIGNeRFModelConfig(NerfactoModelConfig):
    """Configuration for the SIGNeRFModel."""
    _target: Type = field(default_factory=lambda: SIGNeRFModel)
    use_lpips: bool = True
    """Whether to use LPIPS loss"""
    use_l1: bool = True
    """Whether to use L1 loss"""
    patch_size: int = 32
    """Patch size to use for LPIPS loss."""
    lpips_loss_mult: float = 1.0
    """Multiplier for LPIPS loss."""

class SIGNeRFModel(NerfactoModel):
    """Model for SIGNeRF."""

    config: SIGNeRFModelConfig

    def populate_modules(self):
        super().populate_modules()

        if self.config.use_l1:
            self.rgb_loss = L1Loss()
        else:
            self.rgb_loss = MSELoss()
        self.lpips = LearnedPerceptualImagePatchSimilarity()

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> dict:

        loss_dict = {}
        image = batch["image"].to(self.device)
        output = outputs["rgb"]

        loss_dict["rgb_loss"] = self.rgb_loss(image, output)

        if self.config.use_lpips:
            # Before normalization
            # assert torch.isfinite(output).all(), "outputs contains NaN or inf values"
            # assert torch.isfinite(image).all(), "image contains NaN or inf values"

            # Check devices
            # assert output.device == image.device, "Tensors are on different devices"

            # Proceed with normalization
            out_patches = (output.view(-1, self.config.patch_size, self.config.patch_size, 3).permute(0, 3, 1, 2) * 2 - 1).clamp(-1, 1)
            gt_patches = (image.view(-1, self.config.patch_size, self.config.patch_size, 3).permute(0, 3, 1, 2) * 2 - 1).clamp(-1, 1)

            # Before
            loss_dict["lpips_loss"] = self.config.lpips_loss_mult * self.lpips(out_patches, gt_patches)

        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            if self.config.predict_normals:

                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )

        return loss_dict
