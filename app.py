"""
app.py
Advanced Streamlit dashboard for GDP growth prediction with multiple models and interactive visualizations.
"""

import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime
import json

# Local imports
from fetch_data import fetch_worldbank, build_features, get_feature_importance_groups

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🌍 Global GDP Predictor Pro",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Constants
ARTIFACTS_DIR = "artifacts"
MODELS = {
    "RandomForest": "rf_model.pkl",
    "XGBoost": "xgb_model.pkl", 
    "GradientBoosting": "gb_model.pkl",
    "Ridge": "ridge_model.pkl",
    "SVR": "svr_model.pkl",
    "Ensemble": "ensemble_model.pkl"
}

@st.cache_data
def load_data():
    """Load and cache the World Bank data."""
    try:
        df = fetch_worldbank()
        df = build_features(df)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_resource
def load_models():
    """Load all available trained models."""
    models = {}
    for name, filename in MODELS.items():
        model_path = os.path.join(ARTIFACTS_DIR, filename)
        if os.path.exists(model_path):
            try:
                if filename.endswith('.pkl'):
                    models[name] = joblib.load(model_path)
                elif filename.endswith('.h5'):
                    import tensorflow as tf
                    models[name] = tf.keras.models.load_model(model_path)
            except Exception as e:
                st.warning(f"Could not load {name} model: {str(e)}")
    return models

@st.cache_data
def load_metrics():
    """Load model performance metrics."""
    metrics_path = os.path.join(ARTIFACTS_DIR, "all_models_metrics.csv")
    if os.path.exists(metrics_path):
        return pd.read_csv(metrics_path, index_col=0)
    return None

def create_gdp_prediction_chart(country_df, country):
    """Create an interactive GDP prediction chart."""
    fig = go.Figure()
    
    # Actual GDP growth
    fig.add_trace(go.Scatter(
        x=country_df["year"],
        y=country_df["gdp_growth"],
        mode='lines+markers',
        name='Actual GDP Growth',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    # Predicted GDP growth
    fig.add_trace(go.Scatter(
        x=country_df["year"],
        y=country_df["predicted_gdp_growth"],
        mode='lines+markers',
        name='Predicted GDP Growth',
        line=dict(color='#ff7f0e', width=3, dash='dash'),
        marker=dict(size=8, symbol='diamond')
    ))
    
    fig.update_layout(
        title=f"GDP Growth: Actual vs Predicted - {country}",
        xaxis_title="Year",
        yaxis_title="GDP Growth (%)",
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig

def create_feature_importance_chart(model, feature_names, model_name):
    """Create feature importance visualization."""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
    else:
        return None
    
    # Create DataFrame
    feature_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=True).tail(20)
    
    fig = px.bar(
        feature_imp_df,
        x='importance',
        y='feature',
        orientation='h',
        title=f"Top 20 Feature Importance - {model_name}",
        template='plotly_white'
    )
    
    fig.update_layout(height=600)
    return fig

def create_economic_indicators_chart(country_df, country):
    """Create economic indicators visualization."""
    # Select key indicators
    key_indicators = ['cpi_annual_pct', 'domestic_credit_pct_gdp', 'unemployment_rate', 'exports_pct_gdp']
    available_indicators = [col for col in key_indicators if col in country_df.columns]
    
    if not available_indicators:
        return None
    
    fig = make_subplots(
        rows=len(available_indicators), cols=1,
        subplot_titles=[col.replace('_', ' ').title() for col in available_indicators],
        vertical_spacing=0.1
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, indicator in enumerate(available_indicators):
        fig.add_trace(
            go.Scatter(
                x=country_df["year"],
                y=country_df[indicator],
                mode='lines+markers',
                name=indicator.replace('_', ' ').title(),
                line=dict(color=colors[i % len(colors)]),
                showlegend=False
            ),
            row=i+1, col=1
        )
    
    fig.update_layout(
        title=f"Economic Indicators - {country}",
        height=300 * len(available_indicators),
        template='plotly_white',
        showlegend=False
    )
    
    return fig

def create_model_comparison_chart(metrics_df):
    """Create model comparison visualization."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['RMSE (Lower is Better)', 'MAE (Lower is Better)', 
                       'SMAPE (Lower is Better)', 'R² (Higher is Better)'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    metrics = ['rmse', 'mae', 'smape', 'r2']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, metric in enumerate(metrics):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        if metric == 'r2':
            # For R², higher is better, so we'll sort in descending order
            sorted_data = metrics_df.sort_values(metric, ascending=False)
            color_map = colors[i]
        else:
            # For other metrics, lower is better
            sorted_data = metrics_df.sort_values(metric, ascending=True)
            color_map = colors[i]
        
        fig.add_trace(
            go.Bar(
                x=sorted_data.index,
                y=sorted_data[metric],
                marker_color=color_map,
                name=metric.upper(),
                showlegend=False
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title="Model Performance Comparison",
        height=800,
        template='plotly_white',
        showlegend=False
    )
    
    return fig

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🌍 Global GDP Predictor Pro</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced Machine Learning Models for Global Economic Forecasting")
    
    # Load data and models
    with st.spinner("Loading data and models..."):
        df = load_data()
        models = load_models()
        metrics_df = load_metrics()
    
    if df is None:
        st.error("Failed to load data. Please check your data source.")
        return
    
    if not models:
        st.error("No trained models found. Please run the training script first.")
        st.info("Run: `python train.py` to train models")
        return
    
    # Sidebar
    st.sidebar.header("🎛️ Dashboard Controls")
    
    # Model selection
    available_models = list(models.keys())
    selected_model = st.sidebar.selectbox(
        "Select Model",
        available_models,
        index=0
    )
    
    # Country selection
    countries = sorted(df["country"].unique())
    selected_country = st.sidebar.selectbox(
        "Select Country",
        countries,
        index=countries.index("United States") if "United States" in countries else 0
    )
    
    # Year range
    year_range = st.sidebar.slider(
        "Year Range",
        min_value=int(df["year"].min()),
        max_value=int(df["year"].max()),
        value=(int(df["year"].min()), int(df["year"].max()))
    )
    
    # Filter data
    country_df = df[
        (df["country"] == selected_country) & 
        (df["year"] >= year_range[0]) & 
        (df["year"] <= year_range[1])
    ].sort_values("year").copy()
    
    # Get predictions for selected model
    selected_model_obj = models[selected_model]
    feature_cols = [c for c in df.columns if c not in ["country", "countryiso3code", "year", "gdp_growth_next"]]
    X_country = country_df[feature_cols]
    
    # Handle different model types
    if hasattr(selected_model_obj, 'predict'):
        try:
            country_df["predicted_gdp_growth"] = selected_model_obj.predict(X_country)
        except:
            st.error(f"Error making predictions with {selected_model}")
            return
    else:
        st.error(f"Model {selected_model} is not compatible")
        return
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Predictions", "🔍 Model Analysis", "📈 Economic Indicators", 
        "🏆 Model Comparison", "📋 Data Explorer"
    ])
    
    with tab1:
        st.header(f"📊 GDP Growth Predictions - {selected_country}")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            latest_actual = country_df["gdp_growth"].iloc[-1] if len(country_df) > 0 else 0
            st.metric("Latest Actual GDP Growth", f"{latest_actual:.2f}%")
        
        with col2:
            latest_pred = country_df["predicted_gdp_growth"].iloc[-1] if len(country_df) > 0 else 0
            st.metric("Latest Predicted GDP Growth", f"{latest_pred:.2f}%")
        
        with col3:
            if len(country_df) > 1:
                pred_error = abs(latest_actual - latest_pred)
                st.metric("Prediction Error", f"{pred_error:.2f}%")
        
        with col4:
            if len(country_df) > 1:
                accuracy = max(0, 100 - pred_error)
                st.metric("Prediction Accuracy", f"{accuracy:.1f}%")
        
        # Main prediction chart
        if len(country_df) > 0:
            st.plotly_chart(create_gdp_prediction_chart(country_df, selected_country), use_container_width=True)
        
        # Predictions table
        st.subheader("📋 Detailed Predictions")
        display_cols = ["year", "gdp_growth", "predicted_gdp_growth"]
        available_cols = [col for col in display_cols if col in country_df.columns]
        
        if available_cols:
            display_df = country_df[available_cols].copy()
            display_df["gdp_growth"] = display_df["gdp_growth"].round(2)
            display_df["predicted_gdp_growth"] = display_df["predicted_gdp_growth"].round(2)
            st.dataframe(display_df, use_container_width=True)
    
    with tab2:
        st.header("🔍 Model Analysis")
        
        # Model performance metrics
        if metrics_df is not None and selected_model in metrics_df.index:
            st.subheader(f"Performance Metrics - {selected_model}")
            
            col1, col2, col3, col4 = st.columns(4)
            metrics = metrics_df.loc[selected_model]
            
            with col1:
                st.metric("RMSE", f"{metrics['rmse']:.4f}")
            with col2:
                st.metric("MAE", f"{metrics['mae']:.4f}")
            with col3:
                st.metric("SMAPE", f"{metrics['smape']:.2f}%")
            with col4:
                st.metric("R²", f"{metrics['r2']:.4f}")
        
        # Feature importance
        st.subheader("Feature Importance Analysis")
        if hasattr(selected_model_obj, 'feature_importances_') or hasattr(selected_model_obj, 'coef_'):
            feature_imp_chart = create_feature_importance_chart(
                selected_model_obj, feature_cols, selected_model
            )
            if feature_imp_chart:
                st.plotly_chart(feature_imp_chart, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")
    
    with tab3:
        st.header(f"📈 Economic Indicators - {selected_country}")
        
        if len(country_df) > 0:
            indicators_chart = create_economic_indicators_chart(country_df, selected_country)
            if indicators_chart:
                st.plotly_chart(indicators_chart, use_container_width=True)
            
            # Economic indicators summary
            st.subheader("Economic Indicators Summary")
            
            # Get latest values for key indicators
            key_indicators = ['cpi_annual_pct', 'domestic_credit_pct_gdp', 'unemployment_rate', 
                             'exports_pct_gdp', 'industry_pct_gdp', 'agri_pct_gdp']
            
            available_indicators = [col for col in key_indicators if col in country_df.columns]
            
            if available_indicators:
                latest_values = country_df[available_indicators].iloc[-1]
                
                cols = st.columns(len(available_indicators))
                for i, indicator in enumerate(available_indicators):
                    with cols[i]:
                        value = latest_values[indicator]
                        if pd.notna(value):
                            st.metric(
                                indicator.replace('_', ' ').title(),
                                f"{value:.2f}%"
                            )
                        else:
                            st.metric(
                                indicator.replace('_', ' ').title(),
                                "N/A"
                            )
    
    with tab4:
        st.header("🏆 Model Comparison")
        
        if metrics_df is not None:
            # Overall performance comparison
            st.subheader("Overall Model Performance")
            comparison_chart = create_model_comparison_chart(metrics_df)
            st.plotly_chart(comparison_chart, use_container_width=True)
            
            # Detailed metrics table
            st.subheader("Detailed Performance Metrics")
            st.dataframe(metrics_df.round(4), use_container_width=True)
            
            # Best model identification
            best_model = metrics_df['rmse'].idxmin()
            st.success(f"🏆 **Best performing model**: {best_model} (Lowest RMSE: {metrics_df.loc[best_model, 'rmse']:.4f})")
        else:
            st.warning("Model metrics not available. Please run the training script first.")
    
    with tab5:
        st.header("📋 Data Explorer")
        
        # Dataset overview
        st.subheader("Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Countries", len(df["country"].unique()))
        with col2:
            st.metric("Total Years", len(df["year"].unique()))
        with col3:
            st.metric("Total Features", len(feature_cols))
        
        # Data quality
        st.subheader("Data Quality")
        missing_data = df.isnull().sum().sum()
        total_cells = df.shape[0] * df.shape[1]
        data_completeness = ((total_cells - missing_data) / total_cells) * 100
        
        st.metric("Data Completeness", f"{data_completeness:.1f}%")
        
        # Feature categories
        st.subheader("Feature Categories")
        feature_groups = get_feature_importance_groups()
        
        for category, features in feature_groups.items():
            available_features = [f for f in features if f in df.columns]
            if available_features:
                with st.expander(f"{category} ({len(available_features)} features)"):
                    st.write(", ".join(available_features))
        
        # Raw data preview
        st.subheader("Raw Data Preview")
        st.dataframe(df.head(100), use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**🌍 Global GDP Predictor Pro** | Built with Streamlit, Plotly, and Advanced ML | "
        "Data source: World Bank API"
    )

if __name__ == "__main__":
    main()

