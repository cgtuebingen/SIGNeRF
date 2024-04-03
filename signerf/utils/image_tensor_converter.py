""" Module to convert between PIL images and PyTorch tensors """

import torch
from PIL import Image
import numpy as np

def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """ Convert a tensor to an image

    Args:
        tensor (torch.Tensor): tensor to convert

    Returns:
        Image.Image: image
    """

    assert len(tensor.shape) == 3, "Tensor must be of shape (H, W, C)"

    if tensor.shape[2] == 1:
        # Prepare tensor for single channel grayscale image
        np_tensor = tensor.detach().cpu().numpy()
        np_tensor = np_tensor * 255
        np_tensor = np_tensor.astype(np.uint8).squeeze()
        image = Image.fromarray(np_tensor, 'L')
    else:
        # Prepare tensor for RGB image
        assert tensor.shape[2] == 3, "Tensor must be of shape (H, W, 3)"
        np_tensor = tensor.detach().cpu().numpy()
        np_tensor = np_tensor * 255
        np_tensor = np_tensor.astype(np.uint8)
        image = Image.fromarray(np_tensor)

    return image

def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """ Convert an image to a tensor

    Args:
        image (Image.Image): image to convert

    Returns:
        torch.Tensor: tensor
    """

    # Ensure image is RGB not RGBA
    if image.mode == "RGBA":
        image = image.convert("RGB")

    # Convert
    np_tensor = np.array(image, dtype="float32")
    tensor = torch.from_numpy(np_tensor)
    tensor /= 255.0

    return tensor
