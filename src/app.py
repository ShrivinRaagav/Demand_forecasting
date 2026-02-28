import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import shap
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "powergrid_predictions.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "powergrid_lgbm_q50.pkl"
EXPLAINER_PATH = PROJECT_ROOT / "models" / "powergrid_shap_explainer.pkl"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "powergrid_features.csv"

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

    # --- SHAP Explainability Section ---
    st.markdown("---")
    st.subheader(f"🧠 AI Explainability (SHAP) for latest {selected_material} demand")
    
    if EXPLAINER_PATH.exists() and PROCESSED_DATA_PATH.exists():
        try:
            # Load raw features to get column names and a sample line
            raw_features_df = pd.read_csv(PROCESSED_DATA_PATH)
            
            # Find the exact same row in the raw features dataset matching the last prediction
            last_date_str = latest_row['date'].strftime('%Y-%m-%d')
            feature_row = raw_features_df[(raw_features_df['project_id'] == selected_project) & 
                                          (raw_features_df['material'] == selected_material) & 
                                          (raw_features_df['date'] == last_date_str)]
            
            if not feature_row.empty:
                # Drop non-feature columns exactly as model_training.py did
                ignore_cols = ['date', 'project_id', 'location', 'tower_type', 'substation', 'material', 'quantity_demanded']
                features_only = feature_row.drop(columns=[col for col in ignore_cols if col in feature_row.columns])
                
                # Load Explainer
                explainer = joblib.load(EXPLAINER_PATH)
                shap_values = explainer.shap_values(features_only)
                
                # Plot
                fig_shap, ax = plt.subplots(figsize=(10, 5))
                shap.summary_plot(shap_values, features_only, plot_type="bar", show=False, max_display=7)
                plt.title("Top Factors Influencing This Specific Procurement Rule")
                
                col_shap1, col_shap2 = st.columns([2, 1])
                with col_shap1:
                    st.pyplot(fig_shap)
                with col_shap2:
                    st.info("**How to read this:**\nThe longer the bar, the more that specific factor (like the region, or budget variance) pushed the AI to increase or decrease the predicted material drawdown for this week.")
                    
                    # Generate Natural Language Explanation
                    st.write("### AI Conclusion Explanation")
                    
                    # Map feature names to readable business terms
                    feature_map = {
                        'quantity_lag_1w': 'Drawdown last week',
                        'quantity_roll_mean_12w': '12-week consumption trend',
                        'budget_utilization_ratio': 'Budget Burn Rate',
                        'is_monsoon': 'Monsoon Season Status',
                        'tax_rate': 'Prevailing Tax Rate',
                        'location_freq_encoded': 'Geographic Region Profile',
                        'substation_freq_encoded': 'Substation Size/Type',
                        'tower_type_freq_encoded': 'Tower Structure Type'
                    }
                    
                    # Get the individual SHAP values for this specific row
                    row_shap_values = shap_values[0]
                    feature_names = features_only.columns
                    
                    # Sort features by absolute impact
                    impacts = pd.DataFrame({
                        'Feature': feature_names,
                        'Impact': row_shap_values,
                        'Absolute_Impact': np.abs(row_shap_values)
                    }).sort_values('Absolute_Impact', ascending=False)
                    
                    top_positive = impacts[impacts['Impact'] > 0].head(1)
                    top_negative = impacts[impacts['Impact'] < 0].head(1)
                    
                    explanation = f"The AI recommended **{action}** for taking inventory to {forecast_mid:,.0f} units for this specific {selected_project} site.\n\n"
                    
                    if not top_positive.empty:
                        feat_name = feature_map.get(top_positive.iloc[0]['Feature'], top_positive.iloc[0]['Feature'])
                        explanation += f"🔹 **Why Demand is Driven Up**: The primary driver forcing the quantity *higher* right now is the **{feat_name}**.\n\n"
                    
                    if not top_negative.empty:
                        feat_name = feature_map.get(top_negative.iloc[0]['Feature'], top_negative.iloc[0]['Feature'])
                        explanation += f"🔻 **Why Demand is Suppressed**: Conversely, the AI prevented an even higher prediction because the **{feat_name}** is actively suppressing material drawdown."
                        
                    st.success(explanation)
                    
            else:
                st.warning("Could not align the temporal feature row for SHAP calculation.")
                
        except Exception as e:
            st.warning(f"SHAP Explainer Error: {e}")
    else:
        st.info("SHAP models not found. Run model training completely to generate explainer files.")

else:
    st.warning("Prediction data not found. Please run the training pipeline first.")
