
from typing import Union, IO

import base64
import urllib.parse
import io
import functools

from pathlib import Path
from abc import ABC, abstractproperty, abstractmethod
from dataclasses import dataclass
from PIL import Image


class ImageIO(ABC):

    @abstractproperty
    def content(self) -> Image:
        ...


@dataclass
class RawImg(ImageIO):
    image: Image

    @property
    def content(self) -> Image:
        """
        Read PIL.Image
        """        
        return self.image


@dataclass
class EncodedImg(ImageIO):
    content: str

    @property
    def content(self) -> Image:
        image_bytes = base64.b64decode(self.content)
        return Image.open(io.BytesIO(image_bytes))


class FileImg(ImageIO):
    
    def __init__(self, path: Union[str, Path]):
        self._path = path 

    @property
    def content(self) -> Image:
        return Image.open(self._path)



class Img(ImageIO):

    @abstractproperty
    def to_base64(self) -> bytes:
        ...

    @abstractmethod
    def resize_with_proportion(self, max_size):
        ...

    @abstractmethod
    def save(self, file_path: Union[str, io.BufferedWriter]):
        ...        
    

class ImageContent(Img):

    def __init__(self, content: ImageIO):
        self._origin = content

        # if type(content) == str and content.startswith("file://"):
        #     _local_path = urllib.parse.unquote(content[len('file://'):])
        #     self.im_imgg = Image.open(_local_path)
        # # decodes the base64 and parses the image bytes into a
        # # PIL.Image. If the image is not in RGB mode, e.g. for PNGs using a palette,
        # # it will be converted to RGB. This makes sure that they work with
        # # SentenceTransformers/Huggingface Transformers which seems to require a 
        # # (3, height, width) tensor        
        # elif type(content) == str: 
        #     image_bytes = base64.b64decode(content)
        #     self._img = Image.open(io.BytesIO(image_bytes))
        # elif type(content) == bytes:
        #     self._img = Image.open(io.BytesIO(content))
        # else:
        #     self._img = Image.open(content)        

        # if self._img.mode != 'RGB':
        #     self._img = self._img.convert('RGB')      


    @functools.cached_property
    def content(self) -> Image:
        """
        Read PIL.Image
        """        
        return self._origin.content


    @property
    def to_base64(self) -> bytes:
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
        return ImageContent(
            RawImg(_img)
        )
