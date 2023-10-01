
from logging import getLogger
from os import path
from abc import ABC, abstractmethod
from typing import Union, List
from dataclasses import dataclass, field

from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import json
import asyncio
import io
import os

# pylint: disable=E0401

import open_clip
import torch
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer

# pylint: enable=E0401

from fixel.io.img import ImageIO
from fixel.ai.clip.loader import get_models_dir, NAME_MODEL_OPENAI, NAME_PROCESSOR_OPENAI


@dataclass
class ClipInput:
	texts: List[str] = field(default_factory=list)
	images: List[ImageIO] = field(default_factory=list)


@dataclass
class ClipResult:
	text_vectors: list = field(default_factory=list)
	image_vectors: list = field(default_factory=list)


class ClipInference(ABC):
	"""
	Abstract class for Clip Inference models that should be inherited from.
	"""

	@abstractmethod
	def vectorize(self, payload: ClipInput) -> ClipResult:
		...



class ClipInferenceOpenAI(ClipInference):
	"""
	See https://github.com/weaviate/multi2vec-clip-inference/blob/01bb8e5a655061167592e73f6d9b23c979eac0a3/clip.py#L102C10-L102C10
	"""
	clip_model: CLIPModel
	processor: CLIPProcessor
	lock: Lock

	def __init__(self, cuda, cuda_core):
		self.lock = Lock()
		self.device = 'cpu'
		if cuda:
			self.device=cuda_core
		self.clip_model = CLIPModel.from_pretrained(f'${get_models_dir}/${NAME_MODEL_OPENAI}').to(self.device)
		self.processor = CLIPProcessor.from_pretrained(f'${get_models_dir}/${NAME_PROCESSOR_OPENAI}')


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
				image_files = [image.content for image in payload.images]
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






