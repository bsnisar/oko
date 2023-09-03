from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from core.clip import Clip, ClipInput
from PIL import Image
from ratelimit import limits

import pandas as pd
import torch
import requests
import io
import os
import asyncio

# Define transformation to apply to the images (resize, normalize, etc.)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a specific size
    transforms.ToTensor(),           # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])

def resize_with_proportion(image_path, max_size):
    original_image = Image.open(image_path)
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


def _partition_list_into_batches(input_list, batch_size):
    for i in range(0, len(input_list), batch_size):
        yield input_list[i:i + batch_size]


def _list_images(directory_path):
    all_files = os.listdir(directory_path)
    image_files = [file for file in all_files if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    return [os.path.join(directory_path, file) for file in image_files]


async def _inference(clip, batch):
    print(f"Processing {batch}")
    result = await clip.vectorize(
        ClipInput(images=[
            f"file://{os.path.abspath(file_path)}" for file_path in batch
        ])
    )
    print(result.image_vectors)


async def limited_concurrent_tasks(semaphore, clip, batchs):
    tasks = []
    for batch in batchs:
        async with semaphore:
            task = asyncio.create_task(_inference(clip, batch))
            tasks.append(task)
    await asyncio.gather(*tasks)


async def main():
    global clip
    clip = Clip()

    batches = [list(_partition_list_into_batches(_list_images('./__var/unsplash-var'), 25))[0]]

    concurrency_limit = 2  # Adjust the concurrency limit as needed
    semaphore = asyncio.Semaphore(concurrency_limit)

    await limited_concurrent_tasks(semaphore, clip, batches)


asyncio.run(main())




