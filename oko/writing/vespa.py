
import os
from oko.writing.types import ImgDoc, Processor
from vespa.application import Vespa


class VespaProcessor(Processor):

    def __init__(self, url = os.environ.get("VESPA_URL", "http://localhost")):
        self.app = Vespa(url = url, port = 8080)

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
