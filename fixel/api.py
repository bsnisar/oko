from dataclasses import dataclass, field
from fixel.io import ImageIO


@dataclass
class Image():
    uid: str
    url: str
    payload: ImageIO
