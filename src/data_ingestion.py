import os
import sys
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Kaggle dataset information
KAGGLE_DATASET = "c/demand-forecasting-kernels-only"
TRAIN_FILE = RAW_DATA_DIR / "train.csv"

def generate_mock_data():
    """Generates synthetic dataset matching the Kaggle Challenge if API auth fails."""
    print("Generating synthetic dataset (Store Item Demand) for fallback...")
    
    dates = pd.date_range(start="2013-01-01", end="2017-12-31", freq='D')
    stores = range(1, 11)   # 10 stores
    items = range(1, 51)    # 50 items
    
    # Create cartesian product of stores, items, and dates
    # Using a faster method to build the dataframe
    df = pd.MultiIndex.from_product(
        [dates, stores, items], 
        names=['date', 'store', 'item']
    ).to_frame(index=False)
    
    # Generate realistic sales pattern (with seasonality, trend, and noise)
    days_from_start = (df['date'] - df['date'].min()).dt.days
    
    # 1. Base demand per store/item
    np.random.seed(42)
    store_base = {s: np.random.randint(20, 50) for s in stores}
    item_base = {i: np.random.randint(5, 100) for i in items}
    
    base_sales = df['store'].map(store_base) + df['item'].map(item_base)
    
    # 2. Add Yearly seasonality (peaks in summer)
    yearly_seasonality = np.sin((df['date'].dt.dayofyear / 365.25) * 2 * np.pi - np.pi/2) * 20
    
    # 3. Add Weekly seasonality (peaks on weekend)
    weekly_seasonality = df['date'].dt.dayofweek.map({0:0, 1:2, 2:3, 3:5, 4:10, 5:25, 6:30})
    
    # 4. Add slight upward trend
    trend = days_from_start * 0.015
    
    # 5. Add noise
    noise = np.random.normal(0, 5, size=len(df))
    
    # Combine everything
    df['sales'] = np.maximum(base_sales + yearly_seasonality + weekly_seasonality + trend + noise, 0).astype(int)
    
    # Save to CSV
    df.to_csv(TRAIN_FILE, index=False)
    print(f"Synthetic dataset saved successfully at {TRAIN_FILE} ({len(df)} rows)")

def download_kaggle_dataset():
    """
    Downloads the Store Item Demand Forecasting dataset using the Kaggle API.
    Requires having a kaggle.json file.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    if TRAIN_FILE.exists():
        print(f"Dataset already exists at {TRAIN_FILE}")
        return

    print(f"Downloading dataset {KAGGLE_DATASET} to {RAW_DATA_DIR}...")
    
    try:
        # Check if Kaggle token is available (Kaggle CLI exits with 1 if missing)
        result = subprocess.run(
            [sys.executable, "-m", "kaggle", "competitions", "download", "-c", "demand-forecasting-kernels-only", "-p", str(RAW_DATA_DIR)],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            print(f"Kaggle download failed. Reason: {result.stderr.strip() or result.stdout.strip()}")
            generate_mock_data()
        else:
            print("Download successful. Extracting...")
            import zipfile
            zip_file_path = RAW_DATA_DIR / "demand-forecasting-kernels-only.zip"
            if zip_file_path.exists():
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extractall(RAW_DATA_DIR)
                os.remove(zip_file_path)
                print("Extraction complete.")
            
    except Exception as e:
        print(f"An unexpected error occurred during download: {e}")
        generate_mock_data()

if __name__ == "__main__":
    download_kaggle_dataset()
