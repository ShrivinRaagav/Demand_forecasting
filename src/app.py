import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "predictions.csv"

st.set_page_config(page_title="Demand Forecast AI", layout="wide")
st.title("📦 ML-Based Demand Forecasting & Inventory Decisions")

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
    
    stores = df['store'].unique()
    selected_store = st.sidebar.selectbox("Select Store", stores)
    
    items = df['item'].unique()
    selected_item = st.sidebar.selectbox("Select Item", items)
    
    # Filter data
    filtered_df = df[(df['store'] == selected_store) & (df['item'] == selected_item)].copy()
    filtered_df['date'] = pd.to_datetime(filtered_df['date'])
    filtered_df = filtered_df.sort_values('date')
    
    st.subheader(f"Predictive Analytics for Store {selected_store}, Item {selected_item}")
    
    # Key Metrics
    current_stock = filtered_df.iloc[-1]['current_stock']
    forecast_mid = filtered_df.iloc[-1]['forecast_median_50']
    action = filtered_df.iloc[-1]['action']
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Stock", f"{current_stock:.0f}")
    col2.metric("Expected Demand (Median Forecast)", f"{forecast_mid:.0f}")
    
    action_color = "green" if action == "Maintain" else ("red" if action == "Increase" else "orange")
    col3.markdown(f"### Next Action: <span style='color:{action_color}'>{action}</span>", unsafe_allow_html=True)
    
    # Time Series Chart
    st.subheader("Demand Forecast with Uncertainty Intervals (Quantile Regression)")
    
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # Add actual sales
    fig.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['sales'], name='Actual Sales', mode='lines+markers'))
    
    # Add uncertainty band (10th to 90th percentile)
    fig.add_trace(go.Scatter(
        x=filtered_df['date'].tolist() + filtered_df['date'].tolist()[::-1],
        y=filtered_df['forecast_upper_90'].tolist() + filtered_df['forecast_lower_10'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='80% Prediction Interval'
    ))
    
    # Add median forecast
    fig.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['forecast_median_50'], name='Median Forecast', line=dict(color='red', dash='dash')))
    
    fig.update_layout(height=500, xaxis_title="Date", yaxis_title="Demand")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ### System Architecture Insights
    - **Models:** LightGBM Quantile Regressors (predicting 10th, 50th, 90th percentiles).
    - **Uncertainty:** The green band represents the safety range. As long as stock remains within this band, no stockout or excessive overstock is expected.
    - **IEEE XAI:** Under the hood, SHAP values were generated to explain these predictions at a granular feature level.
    """)
    
    st.dataframe(filtered_df[['date', 'sales', 'forecast_lower_10', 'forecast_median_50', 'forecast_upper_90', 'current_stock', 'action']].tail(10))

else:
    st.warning("Prediction data not found. Please run the training pipeline first.")
