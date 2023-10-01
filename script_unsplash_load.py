from PIL import Image
from ratelimit import limits, sleep_and_retry

import pandas as pd
import requests
import io

from content import ImageContent


@sleep_and_retry
@limits(calls=20, period=20)
def _read_photo_image(url) -> ImageContent:
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        return ImageContent(r.content)
    else:
        print(f"GET {url} status_code {r.status_code}")


def _save_file(file_path, data: ImageContent):
    print(f"saving {file_path}")
    with open(file_path, 'wb') as f:
        data.save(f)


df = pd.read_table('./__var/unsplash/photos.tsv000')
for index, row in df.iterrows():

    if index == 25:
        break

    photo_id = row['photo_id']
    photo_image_url = row['photo_image_url']

    img = _read_photo_image(photo_image_url)
    filename = f"{photo_id}__raw.jpg"
    _save_file(
        f'./__var/unsplash-var/{ filename }', 
        img
    ) 

    filename_512 = f"{photo_id}__512x512.jpg"
    _save_file(
        f'./__var/unsplash-var/{ filename_512 }', 
        img.resize_with_proportion(512)
    )


