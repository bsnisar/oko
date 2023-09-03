import weaviate

import os
from logging import getLogger
from fastapi import FastAPI, Response, status
from contextlib import asynccontextmanager

logger = getLogger('uvicorn')

@asynccontextmanager
async def lifespan(app: FastAPI):
	global client

	# client = weaviate.Client("http://localhost:8080") 

	logger.info("Initialization complete")
	yield


app = FastAPI(lifespan=lifespan)


# @app.post("/img/")
# async def add_img(my_file: UploadFile = File(...)):
#     return { "name": my_file.filename }


@app.post("/i/{model}")
async def inference(model: str):
    return "UP"


@app.get("/.well-known/ready")
async def live_and_ready():
    return "UP"

