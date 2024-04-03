""" Module to convert an image to base64 and vice versa """

import base64
from io import BytesIO
from PIL import Image

def image_to_base_64(image: Image.Image) -> bytes:
    """ Convert an image to base64

    Args:
        image (Image.Image): image to convert

    Returns:
        bytes: image in base64 format
    """

    image_bytes = image_to_bytes(image)
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    return image_base64

def base64_to_image(image_base64: bytes) -> Image.Image:
    """ Convert base64 to an image

    Args:
        image_base64 (bytes): image in base64 format

    Returns:
        Image.Image: image
    """

    image_bytes = base64.b64decode(image_base64)
    image = bytes_to_image(image_bytes)
    return image


def image_to_bytes(image: Image.Image) -> bytes:
    """ Convert an image to bytes

    Args:
        image (Image.Image): image to convert

    Returns:
        bytes: image in bytes format
    """

    image_bytes = BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes = image_bytes.getvalue()

    return image_bytes

def bytes_to_image(image_bytes: bytes) -> Image.Image:
    """ Convert bytes to an image

    Args:
        image_bytes (bytes): image in bytes format

    Returns:
        Image.Image: image
    """

    image = Image.open(BytesIO(image_bytes))
    return image