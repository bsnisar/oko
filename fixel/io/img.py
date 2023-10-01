
# pylint: disable=W2301
from typing import Union, IO

import base64
import urllib.parse
import io
import functools

from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass

# pylint: disable=E0401

from PIL import Image

# pylint: enable=E0401


class ImageIO(ABC):
    """
    An abstract base class for handling image content.

    This class defines an abstract property `content` that should be implemented
    by subclasses. It represents the content of an image.

    Attributes:
        content (Image): An abstract property that should return the image content.

    """

    @functools.cached_property
    def content(self) -> Image:
        """
        Get the content of the image.

        Returns:
            Image: The image content.

        """
        return self._content

    @property
    @abstractmethod
    def _content(self) -> Image:
        """
        Get the content of the image.

        Returns:
            Image: The image content.

        """
        ...


    @property
    def to_base64(self) -> bytes:
        """
        Get the content of the image as base64 bytes

        Returns:
            Image as bytes.

        """
        image_bytes = self.content.tobytes()
        base64_image = base64.b64encode(image_bytes)
        return base64_image


    def save(self, file_path: Union[str, io.BufferedWriter]):
        """
        Saves this image under the given filename.  If no format is
        specified, the format to use is determined from the filename
        extension, if possible.
        """
        self.content.save(file_path)


    def resize_with_proportion(self, max_size):
        """
        Resize image with respoct to a proportion.
        """
        original_image = self.content
        width, height = original_image.size

        # Calculate new dimensions while preserving aspect ratio
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))

        # Resize the image
        _img = original_image.resize((new_width, new_height), Image.BILINEAR)
        return create(_img)


@dataclass
class RawImg(ImageIO):
    """
    A class for handling raw image data.

    This class accepts a PIL Image object as input and provides the image's
    content through the `content` property.

    Attributes:
        image (Image): A PIL Image object representing the image.

    """

    image: Image

    @property
    def _content(self) -> Image:
        return self.image


@dataclass
class EncodedImg(ImageIO):
    """
    A class for handling encoded image data.

    This class accepts encoded image data as a string, decodes it, and provides
    the image's content through the `content` property.

    Attributes:
        content (str): Encoded image data as a string.

    """

    content: str

    @property
    def _content(self) -> Image:
        image_bytes = base64.b64decode(self.content)
        return Image.open(io.BytesIO(image_bytes))


class FileImg(ImageIO):
    """
    A class for handling image data from a file.

    This class accepts a file path as input and provides the image's content
    through the `content` property.

    Attributes:
        path (Union[str, Path]): The path to the image file.

    """

    def __init__(self, path: Union[str, Path]):
        self._path = path 

    @property
    def _content(self) -> Image:
        return Image.open(self._path)
   



def create(content: Union[str, io.BytesIO, Image]) -> ImageIO:
    """
    Create an instance of an ImageIO subclass based on the input type.

    Args:
        input (Union[PIL.Image.Image, str, bytes]): The input data for the image.

    Returns:
        ImageIO: An instance of an ImageIO subclass (RawImg, EncodedImg, or FileImg).

    Raises:
        ValueError: If the input type is not recognized.

    """
    if isinstance(content, Image.Image):
        return RawImg(content)
    elif isinstance(content, str):
        path = Path(content)
        if path.exists() and path.is_file():
            return FileImg(path)
        else:
            return EncodedImg(content)
    elif isinstance(content, bytes):
        return EncodedImg(content)
    else:
        raise ValueError("Input type not recognized. Supported types: PIL.Image.Image,"
                        " str (file path), bytes (encoded image data)")
