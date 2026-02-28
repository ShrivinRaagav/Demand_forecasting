# Material Demand Forecasting for Supply Chain Optimization
**Ministry of Power - POWERGRID (Problem Statement ID: 25193)**

## Project Overview
This project presents an uncertainty-aware ML-based material demand forecasting system explicitly designed for PowerGrid infrastructure projects. 

Instead of retail sales, the goal is to plan and predict the exact quantity of goods and materials (e.g., "Transmission Tower Steel", "Conductors", "Transformers") to be procured on a periodic basis to avoid project time delays and budget cost overruns.

Based on factors like **Budgets, Geographic Locations, Tower types, Sub-station types, and Taxes**, the AI estimates future material demand along with confidence ranges (Prediction Intervals) and combines them into actionable procurement orders (Procure/Halt).

## The Pipeline (File by File)

### 1. Data Ingestion (`src/data_ingestion.py`)
- **What it does**: Generates a synthetic but highly complex enterprise PowerGrid dataset mimicking real infrastructure projects. It enforces constraints mathematically mapping multiple `Substation_Types` and `Tower_Types` to distinct locations and applies financial factors like `Taxes`, unit costs, and allocated `Budgets`. 
- **Command**: `python src/data_ingestion.py`

### 2. Supply Chain Preprocessing (`src/preprocessing.py`)
- **What it does**: The AI needs to interpret string locations and project types into mathematics. This script performs Frequency Target Encoding on the infrastructure parameters, and engineered periodic drawdown cycles (e.g., 12-week rolling velocity) to match supply chain temporal dynamics. Finally, it calculates cost-overrun logic features comparing budgets against taxes.
- **Command**: `python src/preprocessing.py`

### 3. AI Modeling & Procurement Engine (`src/model_training.py`)
- **What it does**: Uses **LightGBM Quantile Regression** to estimate bounds. 
  - *10th Percentile*: Absolute minimum material drawdown.
  - *90th Percentile*: Max Safety Buffer.
  By comparing generated "current inventory site levels" against the forecast bands, the system executes an automated Supply Chain Logic script to recommend `Procure Immediately`, `On Schedule`, or `Halt Procurement`. SHAP explainability is implemented natively to show exactly why a prediction was made.
- **Command**: `python src/model_training.py`

### 4. Enterprise PowerGrid Dashboard (`src/app.py`)
- **What it does**: A fully operational Streamlit UI displaying the ML results. A project planner can look at specific project sites and specific materials (e.g., `PRJ_013` -> `Insulators`) and view the exact `Cost Overrun Risk` alongside a visual chart of the current planned schedule against actual drawdowns layered over the 80% Safety Band interval.
- **Command**: `streamlit run src/app.py`

### 5. Adaptive Retraining (`src/adaptive_learning.py`)
- **What it does**: In a multi-year project, constraints change. This script simulates capturing new live Purchase Order (PO) logs and updates the existing decision tree model weights dynamically without memory-intensive retraining.
- **Command**: `python src/adaptive_learning.py`
