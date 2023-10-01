from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from core.clip import Clip, ClipInput
from content import ImageContent

from PIL import Image
from ratelimit import limits

import pandas as pd
import torch
import requests
import io
import os
import asyncio

from pathlib import Path


def _partition_list_into_batches(input_list, batch_size):
    for i in range(0, len(input_list), batch_size):
        yield input_list[i:i + batch_size]


def _list_images(directory_path):
    all_files = os.listdir(directory_path)
    image_files = [
        file for file in all_files 
        if file.lower().endswith(('__512x512.jpg', '__512x512.jpeg', '__512x512.png', '__512x512.gif'))
    ]
    return [os.path.join(directory_path, file) for file in image_files]


async def _inference(clip, batch):
    print(f"Processing {batch}")
    result = await clip.vectorize(
        ClipInput(images=[
            ImageContent(Path(file_path)) for file_path in batch
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

    batches = list(_partition_list_into_batches(
        _list_images('./__var/unsplash-var'), 
        25))
    
    concurrency_limit = 1  # Adjust the concurrency limit as needed
    semaphore = asyncio.Semaphore(concurrency_limit)

    await limited_concurrent_tasks(semaphore, clip, batches)


asyncio.run(main())




