import requests
import time
import numpy as np
import tensorflow as tf
import pandas as pd
import os


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


for index, row in cards.iterrows():
    id = row['id']
    cropped_image = row['card_images'][0]['image_url_cropped']

    if os.path.exists(f"images/{id}.jpg"):
        continue

    # Download the image
    response = requests.get(cropped_image)

    # Check if the image download was successful
    if response.status_code == 200:
        # Save the image to disk
        with open(f"images/{id}.jpg", "wb") as file:
            file.write(response.content)

        # Wait for 1 seconds before downloading the next image
        time.sleep(1)

    else:
        print(f"Failed to download image {id} with status code:", response.status_code)
