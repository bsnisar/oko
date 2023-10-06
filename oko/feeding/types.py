
import os

from logging import getLogger
from dataclasses import dataclass, field
from pydantic import BaseModel
from abc import ABC, abstractmethod
import asyncio

from oko.io import ImageIO
from oko.ai import Inference, VecInput, VecOutput
from oko.ai.clip import Clip

logger = getLogger('inference')

@dataclass
class ImgDoc:
    image_id: str
    image: ImageIO
    image_url: str


class Processor(ABC):

    def __init__(self):
        self._clip: Inference = Processor._create_clip_model()


    @abstractmethod
    async def feed(doc: ImgDoc):
        """
        Abstract method to feed an image document for processing.
        """
        pass


    async def feed_sync(doc: ImgDoc):
        asyncio.run(feed(doc))


    async def _clip_vectorize_image(self, img: ImageIO):
        """
        Vectorize an image using the CLIP model.
        """

        return self._clip.vectorize_image(img)


    @staticmethod
    def _create_clip_model() -> Inference:
        """
        Create and return a CLIP model with optional CUDA support.
        """

        cuda_env = os.getenv("ENABLE_CUDA")
        cuda_support = cuda_env in ["true", "1"]
        cuda_core = os.getenv("CUDA_CORE", "cuda:0")

        if cuda_support:
            logger.info(f"CUDA_CORE set to {cuda_core}")
        else:
            logger.info("Running on CPU")

        return Clip(cuda_support, cuda_core)




class VespaProcessor(Processor):


    async def feed(doc: ImgDoc):
        payload_embed = await self._clip_vectorize_image(doc.payload)
        return {
            "id": doc.image_id,
            "fields": {
                "image_id": doc.image_id,
                "image_url": doc.image_url,
                "image_embed_clip": { "values": payload_embed },
            },
            "create": True,
        }
