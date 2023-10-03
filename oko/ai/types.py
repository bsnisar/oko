
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from oko.io import ImageIO


@dataclass
class VecInput:
	texts: List[str] = field(default_factory=list)
	images: List[ImageIO] = field(default_factory=list)

@dataclass
class VecOutput:
    text_vectors: list = field(default_factory=list)
    image_vectors: list = field(default_factory=list)


class Inference(ABC):
	"""
	Abstract class for Clip Inference models that should be inherited from.
	"""

	async def vectorize_image(self, payload: ImageIO) -> list:
		res = await self.vectorize(VecInput(images=[payload]))
		return res.image_vectors[0]

	@abstractmethod
	async def vectorize(self, payload: VecInput) -> VecOutput:
		...
