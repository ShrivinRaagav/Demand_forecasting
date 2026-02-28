import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "powergrid_predictions.csv"

st.set_page_config(page_title="PowerGrid Procurement AI", layout="wide", page_icon="⚡")

# Header Section
colA, colB = st.columns([1, 4])
with colA:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/5/52/Power_Grid_Corporation_of_India_Logo.svg/1200px-Power_Grid_Corporation_of_India_Logo.svg.png", width=120)
with colB:
    st.title("Material Demand Forecasting & Procurement Optimization")
    st.markdown("**(Ministry of Power - ID 25193)** | *AI-Driven Supply Chain Planning for National Infrastructure Projects*")

# Load predictions
@st.cache_data
def load_data():
    if PREDICTIONS_PATH.exists():
        return pd.read_csv(PREDICTIONS_PATH)
    else:
        return None

df = load_data()

if df is not None:
    st.sidebar.header("Filter Options")
    
    projects = df['project_id'].unique()
    selected_project = st.sidebar.selectbox("Select Project ID", projects)
    
    materials = df['material'].unique()
    selected_material = st.sidebar.selectbox("Select Material Component", materials)
    
    # Filter data
    filtered_df = df[(df['project_id'] == selected_project) & (df['material'] == selected_material)].copy()
    filtered_df['date'] = pd.to_datetime(filtered_df['date'])
    filtered_df = filtered_df.sort_values('date')
    
    current_location = filtered_df.iloc[0]['location']
    
    st.subheader(f"Procurement Analytics for {selected_project} ({current_location}) - {selected_material}")
    
    # Key Metrics (Using the most recent planning week)
    latest_row = filtered_df.iloc[-1]
    
    current_inventory = latest_row['current_inventory']
    forecast_mid = latest_row['forecast_median_50']
    action = latest_row['action']
    projected_overrun = latest_row['projected_overrun']
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Site Inventory", f"{current_inventory:,.0f} units")
    col2.metric("Median Planned Drawdown", f"{forecast_mid:,.0f} units")
    
    if projected_overrun > 0:
        col3.metric("Cost Overrun Risk", f"₹{projected_overrun:,.0f}", delta="Critical", delta_color="inverse")
    else:
        col3.metric("Cost Overrun Risk", "₹0", delta="On Budget", delta_color="normal")
    
    action_color = "green" if action == "On Schedule" else ("red" if action == "Procure Immediately" else "orange")
    col4.markdown(f"### Next Action: <br/><span style='color:{action_color}'>{action}</span>", unsafe_allow_html=True)
    
    # Time Series Chart
    st.markdown("---")
    st.subheader("Materials Demand Forecast with Uncertainty Intervals")
    
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # Add actual material usage
    fig.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['quantity_demanded'], name='Actual Drawdown', mode='lines+markers'))
    
    # Add uncertainty band (10th to 90th percentile)
    fig.add_trace(go.Scatter(
        x=filtered_df['date'].tolist() + filtered_df['date'].tolist()[::-1],
        y=filtered_df['forecast_upper_90'].tolist() + filtered_df['forecast_lower_10'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='80% Prediction Interval (Safety Buffer)'
    ))
    
    # Add median forecast
    fig.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['forecast_median_50'], name='Median Forecast', line=dict(color='red', dash='dash')))
    
    fig.update_layout(height=450, xaxis_title="Timeline (Weekly Planning)", yaxis_title="Quantity Demanded")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ### System Architecture Insights (IEEE Standards)
    - **Models:** LightGBM Quantile Regression models spatial constraints (e.g. North vs South terrain), taxes, and budget vectors.
    - **Risk Handling:** The green band represents the safety range. As long as inventory remains within this band, project delays are avoided while holding costs are restricted.
    - **XAI:** SHAP features run organically during training to highlight whether budget multipliers vs location heavily weighted a material spike.
    """)
    
    st.dataframe(filtered_df[['date', 'quantity_demanded', 'forecast_lower_10', 'forecast_median_50', 'forecast_upper_90', 'current_inventory', 'budget_allocated_inr', 'total_cost_inr', 'action']].tail(10))

else:
    st.warning("Prediction data not found. Please run the training pipeline first.")
