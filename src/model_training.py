import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import shap
import joblib
from pathlib import Path
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "powergrid_features.csv"
MODEL_DIR = PROJECT_ROOT / "models"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "powergrid_predictions.csv"

def load_data():
    logging.info(f"Loading features from {PROCESSED_DATA_PATH}...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    # Needs to be sorted chronologically for valid time series split
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # We will reserve the final 12 weeks of data across all projects as our test set
    test_weeks = 12
    latest_date = df['date'].max()
    split_date = latest_date - pd.DateOffset(weeks=test_weeks)
    
    train_df = df[df['date'] <= split_date].copy()
    test_df = df[df['date'] > split_date].copy()
    
    # Target is quantity demanded for next period
    target = 'quantity_demanded'
    
    # Drop non-predictive features like date and text IDs
    # LightGBM handles categories natively if dtype is 'category', but we specifically 
    # frequency-encoded them in preprocessing, so we can drop the original text.
    ignore_cols = ['date', 'project_id', 'location', 'tower_type', 'substation', 'material', target]
    features = [c for c in df.columns if c not in ignore_cols]
    
    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]
    
    return X_train, y_train, X_test, y_test, features, test_df

def train_quantile_models(X_train, y_train):
    """Train LightGBM models for Procurement Uncertainty."""
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
        joblib.dump(model, MODEL_DIR / f"powergrid_lgbm_q{int(alpha*100)}.pkl")
        
    return models

def evaluate_models(models, X_test, y_test):
    median_model = models['q_50']
    preds = median_model.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    # Calculate MAPE safely (avoid div by zero on zero-demand weeks)
    mask = y_test != 0
    if mask.sum() > 0:
        mape = mean_absolute_percentage_error(y_test[mask], preds[mask])
    else:
        mape = 0.0
        
    logging.info(f"Evaluation Metrics (50th Percentile) - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}")
    return preds

def generate_procurement_decisions(test_df, models, X_test):
    """Implement rule-based Supply Chain Procurement Engine."""
    logging.info("Generating procurement recommendations...")
    results = test_df[['date', 'project_id', 'location', 'material', 'quantity_demanded', 'total_cost_inr', 'budget_allocated_inr']].copy()
    
    # Predict percentiles bounds
    results['forecast_lower_10'] = np.maximum(0, models['q_10'].predict(X_test))
    results['forecast_median_50'] = np.maximum(0, models['q_50'].predict(X_test))
    results['forecast_upper_90'] = np.maximum(0, models['q_90'].predict(X_test))
    
    # Simulate current warehousing inventory (for demonstration purposes)
    np.random.seed(42)
    # Assume we mostly have what was needed, plus/minus some buffer
    current_inventory = results['quantity_demanded'] + np.random.randint(-15, 30, size=len(results))
    results['current_inventory'] = np.maximum(current_inventory, 0)
    
    def decision_logic(row):
        inv = row['current_inventory']
        f_mid = row['forecast_median_50']
        f_up = row['forecast_upper_90']
        
        # Procurement Logic
        if inv < f_mid:
            return "Procure Immediately" # Critical Shortage Risk
        elif f_mid <= inv <= f_up:
            return "On Schedule"         # Perfect
        else:
            return "Halt Procurement"    # Overstocked / Budget Waste
            
    results['action'] = results.apply(decision_logic, axis=1)
    
    # Calculate projected overrun
    results['projected_procurement_cost'] = results['forecast_median_50'] * (results['total_cost_inr'] / np.maximum(results['quantity_demanded'], 1))
    results['projected_overrun'] = np.maximum(0, results['projected_procurement_cost'] - results['budget_allocated_inr'])
    
    logging.info(f"Procurement Actions Distribution:\n{results['action'].value_counts()}")
    
    results.to_csv(PREDICTIONS_PATH, index=False)
    logging.info(f"Saved decisions to {PREDICTIONS_PATH}")
    return results

def shap_explainability(model, X_train):
    """Generate SHAP values for the median model to understand feature importance."""
    logging.info("Calculating SHAP Explainability (using a sample to save time)...")
    sample = X_train.sample(min(1000, len(X_train)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    
    joblib.dump(explainer, MODEL_DIR / "powergrid_shap_explainer.pkl")
    logging.info("SHAP computation complete.")

def plot_sample_forecast(results):
    """Plot the predictions for a sample project and material."""
    project_id = results['project_id'].iloc[0]
    material = results['material'].iloc[0]
    
    logging.info(f"Plotting forecast for Project {project_id}, {material}...")
    sample = results[(results['project_id'] == project_id) & (results['material'] == material)].copy()
    sample = sample.sort_values('date')
    
    plt.figure(figsize=(12, 6))
    
    # Plot Confidence Interval (80% band)
    plt.fill_between(sample['date'], sample['forecast_lower_10'], sample['forecast_upper_90'], 
                     color='green', alpha=0.2, label='80% Prediction Interval (Safety Buffer)')
    
    # Plot Actual vs Median Forecast
    plt.plot(sample['date'], sample['quantity_demanded'], label='Actual Material Drawdown', color='blue', marker='o', markersize=4)
    plt.plot(sample['date'], sample['forecast_median_50'], label='Median Planned Forecast', color='red', linestyle='--')
    
    plt.title(f"Material Demand Forecast vs Actuals ({project_id} - {material})")
    plt.xlabel("Date (Weekly Planning Cycle)")
    plt.ylabel("Quantity Demanded")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    X_train, y_train, X_test, y_test, features, test_df = load_data()
    models = train_quantile_models(X_train, y_train)
    evaluate_models(models, X_test, y_test)
    results = generate_procurement_decisions(test_df, models, X_test)
    
    # Interactive plotting
    plot_sample_forecast(results)
    
    # Explainability
    shap_explainability(models['q_50'], X_train)

if __name__ == "__main__":
    main()
