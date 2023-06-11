from datasets import Dataset
import requests
import time
import numpy as np
import tensorflow as tf
import pandas as pd
import os
from PIL import Image
from io import BytesIO
import base64
from huggingface_hub import HfApi

# API endpoint
url = "https://db.ygoprodeck.com/api/v7/cardinfo.php"

# Request parameters
params = {

}

word_list = dict()
subtype_list = dict()
char_list = dict()
card_number_listed = 0

try:
    response = requests.get(url, params=params)
    data = response.json()

    if response.status_code == 200:
        cards = pd.DataFrame(data=data["data"])
        
except requests.exceptions.RequestException as e:
    print("An error occurred:", e)

# Save the dataframe as csv
cards.to_csv("deck.csv", index=False)

def read_image(file_path):
    with Image.open(file_path) as img:
        return img.convert('RGB')

df_c = cards[["id", "name"]].head(10)
df_c['image'] = df_c.apply(lambda row: f"images/{row['id']}.jpg", axis=1)
df_c['image_data'] = df_c['image'].apply(read_image)

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

df_c["image_data"] = df_c["image_data"].apply(image_to_base64)
huggingface_dataset = Dataset.from_pandas(df_c)

print(df_c.head(10))

# Upload to Huggingface
# huggingface_dataset.push_to_hub("FabioArdi/test", private=True)