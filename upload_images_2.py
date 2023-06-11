import pandas as pd
import os
import shutil
from datasets import load_dataset

df = pd.read_csv('deck.csv')

df_c = df[["id", "name", "frameType"]]
df_c

metadata = []
for index, row in df_c.iterrows():
    dir = f"train/{row['frameType']}"
    if not os.path.exists(dir):
        os.makedirs(dir)

    try:
        sourceImage = f"images/{row['id']}.jpg"
        targetImage = f"{dir}/{row['name']}.jpg"

        shutil.copy(sourceImage, targetImage) 

        metadata.append({
            'file_name': f"{row['frameType']}/{row['name']}.jpg",
            'name': row['name'].replace("\"", ""),
            'frameType': row['frameType']
        })
    except:
        continue

df_m = pd.DataFrame.from_dict(metadata)
df_m.to_csv('train/metadata.csv', index=False)



dataset = load_dataset("imagefolder", data_dir=".")
dataset

dataset.push_to_hub("FabioArdi/test", private=True)

dataset['train'][0]['image']