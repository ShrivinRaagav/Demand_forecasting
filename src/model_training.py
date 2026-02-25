import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import shap
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train_features.csv"
MODEL_DIR = PROJECT_ROOT / "models"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "predictions.csv"

def load_data():
    logging.info(f"Loading features from {PROCESSED_DATA_PATH}...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    # Use last 90 days as test set for time-series validation
    df = df.sort_values('date')
    
    # Simple split
    test_size = 90 * 50 * 10 # 90 days * 50 items * 10 stores
    train_df = df.iloc[:-test_size]
    test_df = df.iloc[-test_size:]
    
    features = [c for c in df.columns if c not in ['date', 'sales']]
    X_train, y_train = train_df[features], train_df['sales']
    X_test, y_test = test_df[features], test_df['sales']
    
    return X_train, y_train, X_test, y_test, features, test_df

def train_quantile_models(X_train, y_train):
    """Train LightGBM models for 10th, 50th (median), and 90th percentiles."""
    models = {}
    alphas = [0.10, 0.50, 0.90]
    
    for alpha in alphas:
        logging.info(f"Training Quantile LightGBM for alpha={alpha}...")
        model = lgb.LGBMRegressor(
            objective='quantile',
            alpha=alpha,
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)
        models[f'q_{int(alpha*100)}'] = model
        
        # Save model
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_DIR / f"lgbm_quantile_{int(alpha*100)}.pkl")
        
    return models

def evaluate_models(models, X_test, y_test):
    """Evaluate point forecast (50th percentile) using standard metrics"""
    median_model = models['q_50']
    preds = median_model.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = mean_absolute_percentage_error(y_test, preds)
    
    logging.info(f"Evaluation Metrics (50th Percentile) - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}")
    return preds

def generate_inventory_decisions(test_df, models, X_test):
    """Implement rule-based Inventory Decision Engine."""
    logging.info("Generating predictions and inventory decisions...")
    results = test_df[['date', 'store', 'item', 'sales']].copy()
    
    # Predict percentiles
    results['forecast_lower_10'] = np.maximum(0, models['q_10'].predict(X_test))
    results['forecast_median_50'] = np.maximum(0, models['q_50'].predict(X_test))
    results['forecast_upper_90'] = np.maximum(0, models['q_90'].predict(X_test))
    
    # Simulate current stock levels (randomly for demonstration, or assume naive previous day sales)
    # We will simulate current stock as actual sales + some random noise to demonstrate the logic appropriately
    np.random.seed(42)
    results['current_stock'] = results['sales'] + np.random.randint(-15, 15, size=len(results))
    results['current_stock'] = np.maximum(results['current_stock'], 0) # ensure stock is not negative
    
    def decision_logic(row):
        c = row['current_stock']
        f_mid = row['forecast_median_50']
        f_up = row['forecast_upper_90']
        
        if c < f_mid:
            return "Increase"
        elif f_mid <= c <= f_up:
            return "Maintain"
        else:
            return "Reduce"
            
    results['action'] = results.apply(decision_logic, axis=1)
    
    logging.info(f"Inventory Actions Distribution:\n{results['action'].value_counts()}")
    
    results.to_csv(PREDICTIONS_PATH, index=False)
    logging.info(f"Saved decisions to {PREDICTIONS_PATH}")
    return results

def shap_explainability(model, X_train):
    """Generate SHAP values for the median model to understand feature importance."""
    logging.info("Calculating SHAP Explainability (using a sample to save time)...")
    sample = X_train.sample(1000, random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    
    # Save the explainer for later use in UI
    joblib.dump(explainer, MODEL_DIR / "shap_explainer.pkl")
    logging.info("SHAP computation complete.")

def main():
    X_train, y_train, X_test, y_test, features, test_df = load_data()
    models = train_quantile_models(X_train, y_train)
    evaluate_models(models, X_test, y_test)
    generate_inventory_decisions(test_df, models, X_test)
    
    # Generate explainability for median model
    shap_explainability(models['q_50'], X_train)

if __name__ == "__main__":
    main()
