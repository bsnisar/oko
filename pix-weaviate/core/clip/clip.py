import open_clip
import torch
import json
import asyncio
import io
import os
import base64
import urllib.parse


from os import path
from abc import ABC, abstractmethod
from typing import Union, List
from PIL import Image
from pydantic import BaseModel, FileUrl
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from logging import getLogger

from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from .pretrained import BASE_DIR

logger = getLogger('clip')


class ClipInput(BaseModel):
	texts: List[str] = []
	images: List[str] = []



class ClipResult:
	text_vectors: list = []
	image_vectors: list = []

	def __init__(self, text_vectors, image_vectors):
		self.text_vectors = text_vectors
		self.image_vectors = image_vectors


class ClipInferenceABS(ABC):
	"""
	Abstract class for Clip Inference models that should be inherited from.
	"""

	@abstractmethod
	def vectorize(self, payload: ClipInput) -> ClipResult:
		...


class ClipInferenceSentenceTransformers(ClipInferenceABS):
	img_model: SentenceTransformer
	text_model: SentenceTransformer
	lock: Lock

	def __init__(self, cuda, cuda_core):
		self.lock = Lock()
		device = 'cpu'
		if cuda:
			device = cuda_core

		self.img_model = SentenceTransformer(f'{BASE_DIR}/clip', device=device)
		self.text_model = SentenceTransformer(f'{BASE_DIR}/text', device=device)

	def vectorize(self, payload: ClipInput) -> ClipResult:
		"""
		Vectorize data from Weaviate.

		Parameters
		----------
		payload : ClipInput
			Input to the Clip model.

		Returns
		-------
		ClipResult
			The result of the model for both images and text.
		"""

		text_vectors = []
		if payload.texts:
			try:
				self.lock.acquire()
				text_vectors = (
					self.text_model
					.encode(payload.texts, convert_to_tensor=True)
					.tolist()
				)
			finally:
				self.lock.release()
		
		image_vectors = []
		if payload.images:
			try:
				self.lock.acquire()
				image_files = [_parse_image(image) for image in payload.images]
				image_vectors = (
					self.img_model
					.encode(image_files, convert_to_tensor=True)
					.tolist()
				)
			finally:
				self.lock.release()

		return ClipResult(
			text_vectors=text_vectors,
			image_vectors=image_vectors,
		)


class ClipInferenceOpenAI(ClipInferenceABS):
	clip_model: CLIPModel
	processor: CLIPProcessor
	lock: Lock

	def __init__(self, cuda, cuda_core):
		self.lock = Lock()
		self.device = 'cpu'
		if cuda:
			self.device=cuda_core
		self.clip_model = CLIPModel.from_pretrained(f'{BASE_DIR}/openai_clip').to(self.device)
		self.processor = CLIPProcessor.from_pretrained(f'{BASE_DIR}/openai_clip_processor')

	def vectorize(self, payload: ClipInput) -> ClipResult:
		"""
		Vectorize data from Weaviate.

		Parameters
		----------
		payload : ClipInput
			Input to the Clip model.

		Returns
		-------
		ClipResult
			The result of the model for both images and text.
		"""

		text_vectors = []
		if payload.texts:
			try:
				self.lock.acquire()
				inputs = self.processor(
					text=payload.texts,
					return_tensors="pt",
					padding=True,
				).to(self.device)

				# Taken from the HuggingFace source code of the CLIPModel
				text_outputs = self.clip_model.text_model(**inputs)
				text_embeds = text_outputs[1]
				text_embeds = self.clip_model.text_projection(text_embeds)

				# normalized features
				text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
				text_vectors = text_embeds.tolist()
			finally:
				self.lock.release()


		image_vectors = []
		if payload.images:
			try:
				self.lock.acquire()
				image_files = [_parse_image(image) for image in payload.images]
				inputs = self.processor(
					images=image_files,
					return_tensors="pt",
					padding=True,
				).to(self.device)

				# Taken from the HuggingFace source code of the CLIPModel
				vision_outputs = self.clip_model.vision_model(**inputs)
				image_embeds = vision_outputs[1]
				image_embeds = self.clip_model.visual_projection(image_embeds)

				# normalized features
				image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
				image_vectors = image_embeds.tolist()
			finally:
				self.lock.release()

		return ClipResult(
			text_vectors=text_vectors,
			image_vectors=image_vectors,
		)


class ClipInferenceOpenCLIP(ClipInferenceABS):
	lock: Lock

	def __init__(self, cuda, cuda_core):
		self.lock = Lock()
		self.device = 'cpu'
		if cuda:
			self.device=cuda_core

		cache_dir = './models/openclip'
		with open(path.join(cache_dir, "config.json")) as user_file:
			config = json.load(user_file)

		model_name = config['model_name']
		pretrained = config['pretrained']

		model, _, preprocess = open_clip.create_model_and_transforms(
			model_name, 
			pretrained=pretrained, 
			cache_dir=cache_dir, 
			device=self.device)
		if cuda:
			model = model.to(device=self.device)

		self.clip_model = model
		self.preprocess = preprocess
		self.tokenizer = open_clip.get_tokenizer(model_name)

	def vectorize(self, payload: ClipInput) -> ClipResult:
		"""
		Vectorize data from Weaviate.

		Parameters
		----------
		payload : ClipInput
			Input to the Clip model.

		Returns
		-------
		ClipResult
			The result of the model for both images and text.
		"""

		text_vectors = []
		if payload.texts:
			try:
				self.lock.acquire()
				with torch.no_grad(), torch.cuda.amp.autocast():
					text = self.tokenizer(payload.texts).to(self.device)
					text_features = self.clip_model.encode_text(text).to(self.device)
					text_features /= text_features.norm(dim=-1, keepdim=True)
				text_vectors = text_features.tolist()
			finally:
				self.lock.release()

		image_vectors = []
		if payload.images:
			try:
				self.lock.acquire()
				image_files = [self.preprocess_image(image) for image in payload.images]
				image_vectors = [self.vectorize_image(image) for image in image_files]
			finally:
				self.lock.release()

		return ClipResult(
			text_vectors=text_vectors,
			image_vectors=image_vectors,
		)

	def preprocess_image(self, base64_encoded_image_string):
		image_bytes = base64.b64decode(base64_encoded_image_string)
		img = Image.open(io.BytesIO(image_bytes))
		return self.preprocess(img).unsqueeze(0).to(device=self.device)

	def vectorize_image(self, image):
		with torch.no_grad(), torch.cuda.amp.autocast():
			image_features = self.clip_model.encode_image(image).to(self.device)
			image_features /= image_features.norm(dim=-1, keepdim=True)

		return image_features.tolist()[0]


class Clip:

	clip: Union[ClipInferenceOpenAI, ClipInferenceSentenceTransformers, ClipInferenceOpenCLIP]
	executor: ThreadPoolExecutor

	def __init__(self, cuda_env = os.getenv("ENABLE_CUDA"), cuda_core = os.getenv("CUDA_CORE")):
		self.executor = ThreadPoolExecutor()

		cuda_support=False

		if cuda_env is not None and cuda_env == "true" or cuda_env == "1":
			cuda_support=True
			if cuda_core is None or cuda_core == "":
				cuda_core = "cuda:0"
			else:
				logger.info("Running on CPU")

		logger.info(f"CUDA support {'on' if cuda_support else 'off'}")

		if path.exists(f'{BASE_DIR}/openai_clip'):
			self.clip = ClipInferenceOpenAI(cuda_support, cuda_core)
		elif path.exists(f'{BASE_DIR}/openclip'):
			self.clip = ClipInferenceOpenCLIP(cuda_support, cuda_core)
		else:
			self.clip = ClipInferenceSentenceTransformers(cuda_support, cuda_core)

	async def vectorize(self, payload: ClipInput):
		"""
		Vectorize data from Weaviate.

		Parameters
		----------
		payload : ClipInput
			Input to the Clip model.

		Returns
		-------
		ClipResult
			The result of the model for both images and text.
		"""

		return await asyncio.wrap_future(self.executor.submit(self.clip.vectorize, payload))


def _resize_with_proportion(original_image, max_size):
    width, height = original_image.size

    # Calculate new dimensions while preserving aspect ratio
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))

    # Resize the image
    resized_image = original_image.resize((new_width, new_height), Image.BILINEAR)
    return resized_image



# _parse_image decodes the base64 and parses the image bytes into a
# PIL.Image. If the image is not in RGB mode, e.g. for PNGs using a palette,
# it will be converted to RGB. This makes sure that they work with
# SentenceTransformers/Huggingface Transformers which seems to require a (3,
# height, width) tensor
def _parse_image(content: str):
	if content.startswith("file://"):
		local_path = urllib.parse.unquote(content[len('file://'):])
		img = Image.open(local_path)
	else:
		image_bytes = base64.b64decode(content)
		img = Image.open(io.BytesIO(image_bytes))

	if img.mode != 'RGB':
		img = img.convert('RGB')


	return _resize_with_proportion(img, 512)