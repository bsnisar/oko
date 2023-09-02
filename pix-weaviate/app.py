import weaviate

import os
from logging import getLogger
from fastapi import FastAPI, Response, status
from contextlib import asynccontextmanager

logger = getLogger('uvicorn')

@asynccontextmanager
async def lifespan(app: FastAPI):
	global client

    # env = os.getenv("WEAVIATE_ENV", default="http://127.0.0.1")
    # env = "http://127.0.0.1"
    logger.info("Running weaviate %s", env)
	client = weaviate.Client("http://127.0.0.1") 

	logger.info("Initialization complete")
	yield


app = FastAPI(lifespan=lifespan)


@app.get("/.well-known/live", response_class=Response)
@app.get("/.well-known/ready", response_class=Response)
async def live_and_ready(response: Response):
	response.status_code = status.HTTP_204_NO_CONTENT


