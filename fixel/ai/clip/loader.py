import os
import sys
import logging
import json

# pylint: disable=E0401

from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
import open_clip

# pylint: enable=E0401

logging.basicConfig(level=logging.INFO)


_MODELS_BASE_DIR = f'{os.getcwd()}/.models/@clip'
_MODELS_VAR_DIR = f'{_MODELS_BASE_DIR}/var'

NAME_MODEL_OPENAI = 'openai_clip'
NAME_PROCESSOR_OPENAI = 'openai_clip_processor'


def get_models_dir():
    return _MODELS_BASE_DIR

def _norm_model_name(p):
    return str(p).replace("/", "_")


def load_openai_clip(
    text_model_name = 'openai/clip-vit-base-patch16',
    clip_model_name = 'openai/clip-vit-base-patch16',
):

    if clip_model_name != text_model_name:
        raise RuntimeError("For OpenAI models the 'CLIP_MODEL_NAME' and 'TEXT_MODEL_NAME' must be the same!")

    logging.info("Downloading OpenAI CLIP model %s from huggingface model hub into [%s]", clip_model_name, get_models_dir())
        
    _model_dir = f'{_MODELS_VAR_DIR}/openai_clip__{_norm_model_name(clip_model_name)}'
    _processor_name = f'{_MODELS_VAR_DIR}/openai_clip_processor__{_norm_model_name(clip_model_name)}'

    clip_model = CLIPModel.from_pretrained(clip_model_name)
    clip_model.save_pretrained(_model_dir)
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    processor.save_pretrained(_processor_name)

    dst = f"{_MODELS_BASE_DIR}/${NAME_MODEL_OPENAI}"
    os.symlink(_model_dir, dst, target_is_directory=True)

    dst = f"{_MODELS_BASE_DIR}/${NAME_PROCESSOR_OPENAI}"
    os.symlink(_processor_name, dst, target_is_directory=True)


def load_huggingface_clip(
    text_model_name = os.getenv('TEXT_MODEL_NAME', 'sentence-transformers/clip-ViT-B-32-multilingual-v1'),
    clip_model_name = os.getenv('CLIP_MODEL_NAME', 'clip-ViT-B-32'),
):

        logging.info("Downloading text model %s from huggingface model hub", text_model_name)
        text_model = SentenceTransformer(text_model_name)
        text_model_dir_ = f'{_MODELS_BASE_DIR}/text__{_norm_model_name(text_model_name)}'
        text_model.save(text_model_dir_)

        logging.info("Downloading img model %s from huggingface model hub", clip_model_name)
        clip_model = SentenceTransformer(clip_model_name)
        model_dir_ = f'{_MODELS_BASE_DIR}/clip__{_norm_model_name(clip_model_name)}'
        clip_model.save(model_dir_)

        os.symlink(text_model_dir_, f"{_MODELS_BASE_DIR}/text", target_is_directory=True)
        os.symlink(model_dir_, f"{_MODELS_BASE_DIR}/clip", target_is_directory=True)
