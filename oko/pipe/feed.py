
import os

from logging import getLogger
from dataclasses import dataclass, field
from pydantic import BaseModel

from oko.io import ImageIO
from oko.ai import Inference, VecInput, VecOutput
from oko.ai.clip import Clip

logger = getLogger('inference')

@dataclass
class ImageDoc:
    image_id: str
    image_url: str
    image: ImageIO


class Processor(object):

    def __init__(self):
        clip: Inference = self._create_clip_model()

    async def feed(doc: ImageDoc):
        return {
            "id": doc.image_id,
            "fields": {
                "image_id": doc.image_id,
                "image_url": doc.image_url,
                "image_embed_clip": { "values": await self.clip.vectorize_image(doc.payload) },
            },
            "create": True,
        }

    def _create_clip_model(self) -> Inference:
        """
        Create model
        """        
        cuda_env = os.getenv("ENABLE_CUDA")
        cuda_support=False
        cuda_core=""        

        if cuda_env is not None and cuda_env == "true" or cuda_env == "1":
            cuda_support=True
            cuda_core = os.getenv("CUDA_CORE")
            if cuda_core is None or cuda_core == "":
                cuda_core = "cuda:0"     
            logger.info(f"CUDA_CORE set to {cuda_core}")
        else:
            logger.info("Running on CPU")

        return Clip(cuda_support, cuda_core)    

