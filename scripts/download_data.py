import os
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

DATASET = "wenruliu/adult-income-dataset"
DOWNLOAD_DIR = Path(r"/home/cloud/ML/data/raw")

def main():
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print("Downloading dataset from Kaggle...")
    api.dataset_download_files(DATASET, path=str(DOWNLOAD_DIR), unzip=True)

    print("Download + unzip complete!")
    print("Files in raw folder:")
    for f in os.listdir(DOWNLOAD_DIR):
        print(" -", f)

if __name__ == "__main__":
    main()

import pandas as pd

df = pd.read_csv(r"/home/cloud/ML/data/raw/adult.csv")
print(df.shape)
print(df.head())
print(df.columns)
