from typing import Annotated

from pydantic import BaseModel, Url, UrlConstraints


FileUrl = Annotated[
    Url, UrlConstraints(allowed_schemes=["file"])
]


class Img(BaseModel):
    pass