import requests
import numpy as np
import tensorflow as tf
import pandas as pd

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

cards()

