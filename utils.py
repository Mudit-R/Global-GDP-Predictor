"""
utils.py
Enhanced utility functions for GDP prediction analysis, visualization, and model evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def load_model(model_name: str, artifacts_dir: str = "artifacts") -> object:
    """
    Load a trained model from the artifacts directory.
    
    Args:
        model_name: Name of the model to load
        artifacts_dir: Directory containing model files
        
    Returns:
        Loaded model object
    """
    model_path = os.path.join(artifacts_dir, f"{model_name.lower()}_model.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        model = joblib.load(model_path)
        print(f"✅ Successfully loaded {model_name} model")
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model {model_name}: {str(e)}")

def load_metrics(artifacts_dir: str = "artifacts") -> pd.DataFrame:
    """
    Load model performance metrics from the artifacts directory.
    
    Args:
        artifacts_dir: Directory containing metrics files
        
    Returns:
        DataFrame with model performance metrics
    """
    metrics_path = os.path.join(artifacts_dir, "all_models_metrics.csv")
    
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    
    try:
        metrics = pd.read_csv(metrics_path, index_col=0)
        print(f"✅ Successfully loaded metrics for {len(metrics)} models")
        return metrics
    except Exception as e:
        raise RuntimeError(f"Error loading metrics: {str(e)}")

def calculate_prediction_confidence(y_true: np.ndarray, y_pred: np.ndarray, 
                                 confidence_level: float = 0.95) -> Dict[str, float]:
    """
    Calculate prediction confidence intervals and uncertainty metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        confidence_level: Confidence level for intervals (default: 0.95)
        
    Returns:
        Dictionary with confidence metrics
    """
    errors = y_true - y_pred
    mse = np.mean(errors**2)
    rmse = np.sqrt(mse)
    
    # Calculate confidence intervals
    z_score = 1.96  # 95% confidence level
    margin_of_error = z_score * rmse
    
    confidence_interval = {
        'lower_bound': y_pred - margin_of_error,
        'upper_bound': y_pred + margin_of_error
    }
    
    # Coverage rate (percentage of true values within confidence interval)
    within_interval = np.sum((y_true >= confidence_interval['lower_bound']) & 
                            (y_true <= confidence_interval['upper_bound']))
    coverage_rate = (within_interval / len(y_true)) * 100
    
    return {
        'rmse': rmse,
        'margin_of_error': margin_of_error,
        'confidence_level': confidence_level * 100,
        'coverage_rate': coverage_rate,
        'confidence_interval': confidence_interval
    }

def create_residual_analysis_plot(y_true: np.ndarray, y_pred: np.ndarray, 
                                title: str = "Residual Analysis") -> go.Figure:
    """
    Create comprehensive residual analysis plots.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        
    Returns:
        Plotly figure with residual analysis
    """
    residuals = y_true - y_pred
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Residuals vs Predicted', 'Residuals Distribution', 
                       'Q-Q Plot', 'Residuals vs Index'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Residuals vs Predicted
    fig.add_trace(
        go.Scatter(
            x=y_pred, y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(color='blue', size=6, opacity=0.6)
        ),
        row=1, col=1
    )
    
    # Add horizontal line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    
    # Residuals Distribution
    fig.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=30,
            name='Residuals Distribution',
            marker_color='lightblue'
        ),
        row=1, col=2
    )
    
    # Q-Q Plot (simplified)
    sorted_residuals = np.sort(residuals)
    theoretical_quantiles = np.percentile(sorted_residuals, np.linspace(0, 100, len(sorted_residuals)))
    
    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles, y=sorted_residuals,
            mode='markers',
            name='Q-Q Plot',
            marker=dict(color='green', size=6)
        ),
        row=2, col=1
    )
    
    # Add diagonal line for perfect normal distribution
    min_val = min(theoretical_quantiles.min(), sorted_residuals.min())
    max_val = max(theoretical_quantiles.max(), sorted_residuals.max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines',
            name='Perfect Normal',
            line=dict(color='red', dash='dash'),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Residuals vs Index
    fig.add_trace(
        go.Scatter(
            x=list(range(len(residuals))), y=residuals,
            mode='lines+markers',
            name='Residuals vs Index',
            marker=dict(color='orange', size=6)
        ),
        row=2, col=2
    )
    
    # Add horizontal line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)
    
    fig.update_layout(
        title=title,
        height=800,
        template='plotly_white',
        showlegend=False
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Predicted Values", row=1, col=1)
    fig.update_yaxes(title_text="Residuals", row=1, col=1)
    fig.update_xaxes(title_text="Residuals", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=1)
    fig.update_yaxes(title_text="Sample Quantiles", row=2, col=1)
    fig.update_xaxes(title_text="Index", row=2, col=2)
    fig.update_yaxes(title_text="Residuals", row=2, col=2)
    
    return fig

def create_feature_correlation_heatmap(df: pd.DataFrame, 
                                     feature_cols: List[str],
                                     title: str = "Feature Correlation Matrix") -> go.Figure:
    """
    Create a correlation heatmap for features.
    
    Args:
        df: DataFrame containing the data
        feature_cols: List of feature column names
        title: Plot title
        
    Returns:
        Plotly figure with correlation heatmap
    """
    # Calculate correlation matrix
    correlation_matrix = df[feature_cols].corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(correlation_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        height=600,
        template='plotly_white',
        xaxis_title="Features",
        yaxis_title="Features"
    )
    
    return fig

def create_time_series_decomposition(df: pd.DataFrame, 
                                   country: str,
                                   value_col: str = "gdp_growth",
                                   period: int = 5) -> go.Figure:
    """
    Create time series decomposition plot for trend, seasonal, and residual components.
    
    Args:
        df: DataFrame with time series data
        country: Country name for filtering
        value_col: Column name for the value to decompose
        period: Period for seasonal decomposition
        
    Returns:
        Plotly figure with decomposition components
    """
    # Filter data for specific country
    country_data = df[df['country'] == country].sort_values('year')
    
    if len(country_data) < period * 2:
        raise ValueError(f"Insufficient data for decomposition. Need at least {period * 2} data points.")
    
    # Simple moving average for trend
    trend = country_data[value_col].rolling(window=period, center=True).mean()
    
    # Detrended data
    detrended = country_data[value_col] - trend
    
    # Simple seasonal component (using period)
    seasonal = detrended.groupby(country_data['year'] % period).mean()
    seasonal_component = country_data['year'].map(lambda x: seasonal[x % period])
    
    # Residuals
    residuals = detrended - seasonal_component
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=['Original Data', 'Trend', 'Seasonal', 'Residuals'],
        vertical_spacing=0.05
    )
    
    # Original data
    fig.add_trace(
        go.Scatter(
            x=country_data['year'],
            y=country_data[value_col],
            mode='lines+markers',
            name='Original Data',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    # Trend
    fig.add_trace(
        go.Scatter(
            x=country_data['year'],
            y=trend,
            mode='lines',
            name='Trend',
            line=dict(color='red', width=2)
        ),
        row=2, col=1
    )
    
    # Seasonal
    fig.add_trace(
        go.Scatter(
            x=country_data['year'],
            y=seasonal_component,
            mode='lines+markers',
            name='Seasonal',
            line=dict(color='green')
        ),
        row=3, col=1
    )
    
    # Residuals
    fig.add_trace(
        go.Scatter(
            x=country_data['year'],
            y=residuals,
            mode='lines+markers',
            name='Residuals',
            line=dict(color='orange')
        ),
        row=4, col=1
    )
    
    fig.update_layout(
        title=f"Time Series Decomposition - {country} ({value_col})",
        height=800,
        template='plotly_white',
        showlegend=False
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text=value_col.replace('_', ' ').title(), row=1, col=1)
    fig.update_yaxes(title_text="Trend", row=2, col=1)
    fig.update_yaxes(title_text="Seasonal", row=3, col=1)
    fig.update_yaxes(title_text="Residuals", row=4, col=1)
    
    return fig

def generate_model_report(model_name: str, 
                         metrics: Dict[str, float],
                         feature_importance: Optional[pd.DataFrame] = None,
                         artifacts_dir: str = "artifacts") -> str:
    """
    Generate a comprehensive model performance report.
    
    Args:
        model_name: Name of the model
        metrics: Dictionary of performance metrics
        feature_importance: DataFrame with feature importance (optional)
        artifacts_dir: Directory to save the report
        
    Returns:
        Path to the generated report
    """
    report_content = f"""
# Model Performance Report: {model_name}

## Performance Metrics
- **RMSE**: {metrics.get('rmse', 'N/A'):.4f}
- **MAE**: {metrics.get('mae', 'N/A'):.4f}
- **SMAPE**: {metrics.get('smape', 'N/A'):.2f}%
- **R²**: {metrics.get('r2', 'N/A'):.4f}

## Model Information
- **Model Type**: {model_name}
- **Training Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Dataset**: World Bank Economic Indicators

## Performance Analysis
"""
    
    # Add performance analysis
    if metrics.get('r2', 0) > 0.7:
        report_content += "- **Excellent Performance**: R² > 0.7 indicates strong predictive power\n"
    elif metrics.get('r2', 0) > 0.5:
        report_content += "- **Good Performance**: R² > 0.5 indicates reasonable predictive power\n"
    else:
        report_content += "- **Moderate Performance**: R² < 0.5 indicates room for improvement\n"
    
    if feature_importance is not None:
        report_content += "\n## Top 10 Most Important Features\n"
        top_features = feature_importance.head(10)
        for idx, row in top_features.iterrows():
            report_content += f"- {row['feature']}: {row['importance']:.4f}\n"
    
    # Save report
    report_path = os.path.join(artifacts_dir, f"{model_name.lower()}_report.md")
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f"📄 Model report saved to: {report_path}")
    return report_path

def validate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, bool]:
    """
    Validate prediction results for common issues.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {}
    
    # Check for NaN values
    validation_results['no_nan_predictions'] = not np.any(np.isnan(y_pred))
    validation_results['no_nan_actuals'] = not np.any(np.isnan(y_true))
    
    # Check for infinite values
    validation_results['no_inf_predictions'] = not np.any(np.isinf(y_pred))
    validation_results['no_inf_actuals'] = not np.any(np.isinf(y_true))
    
    # Check prediction range (GDP growth typically between -50% and +50%)
    validation_results['reasonable_range'] = np.all((y_pred >= -50) & (y_pred <= 50))
    
    # Check for constant predictions
    validation_results['not_constant'] = np.std(y_pred) > 1e-6
    
    # Check prediction vs actual correlation
    if len(y_true) > 1:
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        validation_results['positive_correlation'] = correlation > 0
    
    return validation_results
