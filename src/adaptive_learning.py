import pandas as pd
import lightgbm as lgb
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train_features.csv"

def adaptive_retraining(new_data_path=None):
    """
    Demonstrates Adaptive Learning workflow.
    Takes new incoming data, and continues training the existing model (Fine-tuning).
    """
    logging.info("Starting Adaptive Retraining Pipeline...")
    
    # In a real scenario, this would load specifically the *newly recorded* data
    # Here, we simulate by just randomly sampling from the existing data to represent "new" days
    df = pd.read_csv(PROCESSED_DATA_PATH)
    new_data = df.sample(frac=0.05, random_state=99) # Simulate 5% new data
    
    features = [c for c in df.columns if c not in ['date', 'sales']]
    X_new, y_new = new_data[features], new_data['sales']
    
    alphas = [10, 50, 90]
    
    for alpha in alphas:
        model_path = MODEL_DIR / f"lgbm_quantile_{alpha}.pkl"
        if not model_path.exists():
            logging.error(f"Cannot find existing model at {model_path}. Train baseline first.")
            return
            
        logging.info(f"Loading existing Alpha {alpha} model...")
        existing_model = joblib.load(model_path)
        
        # Continue training with LightGBM's init_model functionality
        # LightGBM requires passing the booster to init_model
        # Note: scikit-learn API of lightgbm allows passing `init_model` in fit
        logging.info("Fine-tuning model on new incoming data...")
        existing_model.fit(
            X_new, y_new,
            init_model=existing_model.booster_  # The booster of the existing model
        )
        
        # Overwrite or save as new version
        new_model_path = MODEL_DIR / f"lgbm_quantile_{alpha}_v2.pkl"
        joblib.dump(existing_model, new_model_path)
        logging.info(f"Updated model saved to {new_model_path}")

if __name__ == "__main__":
    adaptive_retraining()
