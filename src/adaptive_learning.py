import pandas as pd
import lightgbm as lgb
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "powergrid_features.csv"

def adaptive_retraining():
    """
    Demonstrates Adaptive Learning workflow for PowerGrid.
    Takes new incoming Purchase Orders/Material data and fine-tunes the trees.
    """
    logging.info("Starting Adaptive Retraining Pipeline...")
    
    df = pd.read_csv(PROCESSED_DATA_PATH)
    new_data = df.sample(frac=0.05, random_state=99) # Simulate new construction site logs
    
    target = 'quantity_demanded'
    ignore_cols = ['date', 'project_id', 'location', 'tower_type', 'substation', 'material', target]
    features = [c for c in df.columns if c not in ignore_cols]
    
    X_new, y_new = new_data[features], new_data[target]
    
    alphas = [10, 50, 90]
    
    for alpha in alphas:
        model_path = MODEL_DIR / f"powergrid_lgbm_q{alpha}.pkl"
        if not model_path.exists():
            logging.error(f"Cannot find existing model {model_path}. Train baseline first.")
            return
            
        logging.info(f"Loading existing Alpha {alpha} supply chain model...")
        existing_model = joblib.load(model_path)
        
        logging.info("Fine-tuning model on newly captured PO histories...")
        existing_model.fit(
            X_new, y_new,
            init_model=existing_model.booster_  
        )
        
        new_model_path = MODEL_DIR / f"powergrid_lgbm_q{alpha}_v2.pkl"
        joblib.dump(existing_model, new_model_path)
        logging.info(f"Updated weights merged tracking new Budget/Tax profiles. Saved to {new_model_path}")

if __name__ == "__main__":
    adaptive_retraining()
