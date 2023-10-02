
import os
from logging import getLogger
from dataclasses import dataclass, field

from fixel.io import ImageIO
from fixel.ai.clip import Clip

logger = getLogger('inference')

@dataclass
class FeedImage():
    uid: str
    url: str
    payload: ImageIO



class InferencePipeline:

    def __init__(self):
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

        self.clip = Clip(cuda_support, cuda_core)


    def inference_multi2vec(self, records: list[FeedImage]):
        pass

    def feed(self):
        pass

    def run_pipeline(self):
        pass
