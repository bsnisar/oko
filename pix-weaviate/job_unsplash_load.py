from PIL import Image
from ratelimit import limits, sleep_and_retry

import pandas as pd
import requests
import io


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


@sleep_and_retry
@limits(calls=20, period=20)
def read_photo_image(url):
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        return io.BytesIO(r.content)
    else:
        print(f"GET {url} status_code {r.status_code}")


def save_file(file_path, bytes_io):
    print(f"saving {file_path}")
    with open(file_path, 'wb') as f:
        f.write(bytes_io.getvalue())


df = pd.read_table('./__var/unsplash/photos.tsv000')
for index, row in df.iterrows():
    photo_id = row['photo_id']
    photo_image_url = row['photo_image_url']
    bytes_io = read_photo_image(photo_image_url)
    filename = f"{photo_id}__xl.jpg"
    save_file(f'./__var/unsplash-var/{ filename }', bytes_io) 


