import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_FILE = RAW_DATA_DIR / "powergrid_demand_data.csv"

def generate_powergrid_data():
    """Generates synthetic dataset constraint-mapped for Ministry of Power (Problem ID 25193)."""
    logging.info("Generating POWERGRID Supply Chain dataset...")
    
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    if TRAIN_FILE.exists():
        logging.info(f"Dataset already exists at {TRAIN_FILE}. Skipping generation.")
        return
        
    np.random.seed(42)
    
    # 1. Define Constraints from Problem Statement
    dates = pd.date_range(start="2020-01-01", end="2024-12-31", freq='W-MON') # Weekly planning frequency
    project_locations = ['North', 'South', 'East', 'West', 'North-East']
    tower_types = ['Suspension Tower', 'Tension Tower', 'Transposition Tower', 'Special Tower']
    substation_types = ['220kV', '400kV', '765kV', 'HVDC']
    materials = ['Conductor', 'Insulator', 'Transmission Tower Steel', 'Transformer', 'Circuit Breaker']
    
    # Create projects combinations
    projects = []
    for loc in project_locations:
        for twr in tower_types:
            for sub in substation_types:
                projects.append({'location': loc, 'tower_type': twr, 'substation': sub})
                
    # Select subset of active projects
    active_projects = pd.DataFrame(projects).sample(n=20, random_state=42).reset_index(drop=True)
    active_projects['project_id'] = [f"PRJ_{i:03d}" for i in range(1, len(active_projects)+1)]
    
    # 2. Build the Time Series Dataframe
    logging.info("Building base multi-index dataframe...")
    df_list = []
    for _, prj in active_projects.iterrows():
        for mat in materials:
            temp_df = pd.DataFrame({'date': dates})
            temp_df['project_id'] = prj['project_id']
            temp_df['location'] = prj['location']
            temp_df['tower_type'] = prj['tower_type']
            temp_df['substation'] = prj['substation']
            temp_df['material'] = mat
            df_list.append(temp_df)
            
    df = pd.concat(df_list, ignore_index=True)
    
    # 3. Apply Domain Logic to generate 'Quantity Demanded'
    logging.info("Applying domain logic to generate demand, budget, and tax calculations...")
    
    # Base material multipliers based on infrastructure sizes
    materials_base = {
        'Transmission Tower Steel': 500,
        'Conductor': 1200,
        'Insulator': 800,
        'Transformer': 5,
        'Circuit Breaker': 15
    }
    
    substation_mult = {'220kV': 1.0, '400kV': 2.5, '765kV': 5.0, 'HVDC': 8.0}
    tower_mult = {'Suspension Tower': 1.0, 'Tension Tower': 1.5, 'Transposition Tower': 1.8, 'Special Tower': 3.0}
    
    # Calculate base mean demand per row
    df['base_demand'] = df['material'].map(materials_base) * \
                        df['substation'].map(substation_mult) * \
                        df['tower_type'].map(tower_mult)
                        
    # Apply yearly seasonality (construction peaks pre-monsoon and post-monsoon)
    # Monsoon in India: June-Sept (~week 22 to ~week 39) -> demand drops
    week_of_year = df['date'].dt.isocalendar().week
    seasonality = np.where((week_of_year >= 22) & (week_of_year <= 39), 0.5, 1.2)
    
    # Apply long term growth trend (PowerGrid expanding)
    days_from_start = (df['date'] - df['date'].min()).dt.days
    trend = 1 + (days_from_start * 0.0005)
    
    # Noise and actual demand
    noise = np.random.normal(1.0, 0.2, size=len(df))
    df['quantity_demanded'] = np.maximum(df['base_demand'] * seasonality * trend * noise, 0).astype(int)
    
    # 4. Generate Budget and Tax constraints
    # Tax rate fluctuations (e.g. GST changes)
    df['tax_rate'] = np.where(df['date'].dt.year < 2022, 18.0, 18.5) # Slight tax increase in 2022
    
    # Material unit costs (synthetic)
    unit_cost = {
        'Transmission Tower Steel': 80000, # INR per MT
        'Conductor': 120000,              # INR per km
        'Insulator': 5000,                 # INR per unit
        'Transformer': 50000000,         # INR per unit
        'Circuit Breaker': 1500000        # INR per unit
    }
    df['unit_cost_inr'] = df['material'].map(unit_cost)
    
    # Total Cost = Quantity * Unit Cost * (1 + Tax)
    df['total_cost_inr'] = df['quantity_demanded'] * df['unit_cost_inr'] * (1 + (df['tax_rate']/100))
    
    # Budget Allocated (Usually slightly lower or higher than actual cost, creating overrun risks)
    budget_variance = np.random.normal(1.05, 0.15, size=len(df)) # Budget is sometimes higher, sometimes lower
    df['budget_allocated_inr'] = df['total_cost_inr'] * budget_variance
    
    # Clean up columns
    df = df.drop(columns=['base_demand'])
    
    # Sort and save
    df = df.sort_values(by=['project_id', 'material', 'date'])
    df.to_csv(TRAIN_FILE, index=False)
    logging.info(f"POWERGRID dataset generated successfully at {TRAIN_FILE} ({len(df)} rows)")

if __name__ == "__main__":
    generate_powergrid_data()
