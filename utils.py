"""
utils.py

Helper functions for plotting, model loading, prediction and SHAP explanations.
"""
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import shap
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")

def load_artifacts():
    rf = joblib.load(ARTIFACTS_DIR / "rf_model.pkl")
    xgb = joblib.load(ARTIFACTS_DIR / "xgb_model.pkl")
    feature_cols = joblib.load(ARTIFACTS_DIR / "feature_cols.pkl")
    return {"rf": rf, "xgb": xgb, "feature_cols": feature_cols}

def predict(model, X):
    return model.predict(X)

def shap_explain(model, X_sample, feature_names):
    """Return SHAP values (as numpy array) and expected value."""
    # Use TreeExplainer for tree models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    expected_value = explainer.expected_value
    return shap_values, expected_value

def plot_time_series(df, country, indicator):
    sub = df[df["country"] == country].sort_values("year")
    fig = px.line(sub, x="year", y=indicator, title=f"{indicator} — {country}")
    return fig
