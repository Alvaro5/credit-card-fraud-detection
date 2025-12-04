import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

DATA_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(DATA_DIR, "creditcard.csv")

def download_creditcard_data():

    if os.path.exists(CSV_PATH):
        return CSV_PATH

    api = KaggleApi()
    api.authenticate()

    dataset = "mlg-ulb/creditcardfraud"
    print("Downloading creditcard fraud dataset...")
    api.dataset_download_files(dataset, path=DATA_DIR, unzip=True)

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError("Dataset downloaded but creditcard.csv not found")

    return CSV_PATH

def load_creditcard_df():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            "creditcard.csv not found. Run download_creditcard_data() first."
        )
    return pd.read_csv(CSV_PATH)
