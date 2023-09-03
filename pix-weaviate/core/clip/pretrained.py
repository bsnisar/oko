import os
import sys
import logging
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
import open_clip
import json

logging.basicConfig(level=logging.INFO)

BASE_DIR = f'{os.getcwd()}/__models/@clip'

def _norm_model_name(p):
    return str(p).replace("/", "_")


def download(
        text_model_name = os.getenv('TEXT_MODEL_NAME', 'sentence-transformers/clip-ViT-B-32-multilingual-v1'),
        clip_model_name = os.getenv('CLIP_MODEL_NAME', 'clip-ViT-B-32'),
        clip_model_type = os.getenv('CLIP_MODEL_TYPE', 'sentencetransformer'),
        open_clip_model_name = os.getenv('OPEN_CLIP_MODEL_NAME'),
        open_clip_pretrained = os.getenv('OPEN_CLIP_PRETRAINED')
):

    if open_clip_model_name is not None \
        and open_clip_model_name != "" \
        and open_clip_pretrained is not None \
        and open_clip_pretrained != "":

        def check_model_and_pretrained(model_name: str, pretrained: str):
            if (model_name, pretrained) in open_clip.list_pretrained():
                return

            logging.error("Fatal: Available pairs are:")
            for pair in open_clip.list_pretrained():
                logging.error(f"Fatal: model: {pair[0]} pretrained: {pair[1]}")
            
            raise RuntimeError(f"Fatal: Match not found for OPEN_CLIP model {model_name} with pretrained {pretrained} pair")

        logging.info(f"Checking if OPEN_CLIP model {open_clip_model_name} and pretrained {open_clip_pretrained} is a valid pair")
        check_model_and_pretrained(open_clip_model_name, open_clip_pretrained)

        cache_dir = f"{BASE_DIR}/openclip__{_norm_model_name(open_clip_model_name)}"

        logging.info(f"Downloading OPEN_CLIP model {open_clip_model_name} with {open_clip_pretrained} to {cache_dir}")
        model, _, preprocess = open_clip.create_model_and_transforms(
            open_clip_model_name,
            pretrained=open_clip_pretrained,
            cache_dir=cache_dir
        )
        model_config = open_clip.get_model_config(open_clip_model_name)

        config = {
            "model_name": open_clip_model_name,
            "pretrained": open_clip_pretrained,
            "model_config": model_config,
            "cache_dir": cache_dir
        }

        with open(os.path.join(cache_dir, "config.json"), 'w') as f:
            json.dump(config, f)

        os.symlink(cache_dir, f"{BASE_DIR}/openclip", target_is_directory=True)
        logging.info("Successfully downloaded and validated model and pretrained")

    if text_model_name is None or text_model_name == "":
        raise RuntimeError("Fatal: TEXT_MODEL_NAME is required")

    if clip_model_name is None or clip_model_name == "":
        raise RuntimeError("Fatal: CLIP_MODEL_NAME is required")

    if clip_model_name.startswith('openai/') or clip_model_type.lower() == "openai":
        if clip_model_name != text_model_name:
            raise RuntimeError("For OpenAI models the 'CLIP_MODEL_NAME' and 'TEXT_MODEL_NAME' must be the same!")

        logging.info("Downloading OpenAI CLIP model %s from huggingface model hub", clip_model_name)
        
        _model_dir = f'{BASE_DIR}/openai_clip__{_norm_model_name(clip_model_name)}'
        _processor_name = f'{BASE_DIR}/openai_clip_processor__{_norm_model_name(clip_model_name)}'

        clip_model = CLIPModel.from_pretrained(clip_model_name)
        clip_model.save_pretrained(_model_dir)
        processor = CLIPProcessor.from_pretrained(clip_model_name)
        processor.save_pretrained(_processor_name)

        dst = f"{BASE_DIR}/openai_clip"
        os.symlink(_model_dir, dst, target_is_directory=True)

        dst = f"{BASE_DIR}/openai_clip_processor"
        os.symlink(_processor_name, dst, target_is_directory=True)

    else:
        logging.info("Downloading text model %s from huggingface model hub", text_model_name)
        text_model = SentenceTransformer(text_model_name)
        text_model_dir_ = f'{BASE_DIR}/text__{_norm_model_name(text_model_name)}'
        text_model.save(text_model_dir_)

        logging.info("Downloading img model %s from huggingface model hub", clip_model_name)
        clip_model = SentenceTransformer(clip_model_name)
        model_dir_ = f'{BASE_DIR}/clip__{_norm_model_name(clip_model_name)}'
        clip_model.save(model_dir_)

        os.symlink(text_model_dir_, f"{BASE_DIR}/text", target_is_directory=True)
        os.symlink(model_dir_, f"{BASE_DIR}/clip", target_is_directory=True)