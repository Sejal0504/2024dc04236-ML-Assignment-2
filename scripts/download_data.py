import os
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd

# Path
BASE_DIR = Path(__file__).resolve().parent.parent
DOWNLOAD_DIR = BASE_DIR / "data" / "raw"

DATASET = "wenruliu/adult-income-dataset"


def main():

   
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Authenticate Kaggle
    api = KaggleApi()
    api.authenticate()

    print("Downloading dataset from Kaggle...")
    api.dataset_download_files(DATASET, path=str(DOWNLOAD_DIR), unzip=True)

    print("Download + unzip complete!")

    print("\nFiles in raw folder:")
    for f in os.listdir(DOWNLOAD_DIR):
        print(" -", f)

    # Load dataset to verify
    file_path = DOWNLOAD_DIR / "adult.csv"

    if file_path.exists():
        df = pd.read_csv(file_path)

        print("\nDataset loaded successfully!")
        print("Shape:", df.shape)

        print("\nFirst 5 rows:")
        print(df.head())

        print("\nColumns:")
        print(df.columns)

    else:
        print("\nERROR: adult.csv not found in", DOWNLOAD_DIR)


if __name__ == "__main__":
    main()

