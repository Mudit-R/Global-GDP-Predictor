"""
app.py
Streamlit dashboard for GDP growth prediction and comparison.
"""

import os
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from fetch_data import fetch_worldbank, build_features

ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "rf_model.pkl")

st.set_page_config(page_title="Global GDP Predictor", layout="centered")

# Load model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# Load dataset
@st.cache_data
def load_data():
    df = fetch_worldbank()
    df = build_features(df)
    return df

def main():
    st.title("🌍 Global GDP Growth Predictor")
    st.write("Predict and compare GDP growth using World Bank indicators.")

    # Load model + data
    model = load_model()
    df = load_data()

    countries = sorted(df["country"].unique())
    country = st.selectbox("Select a country", countries)

    # Subset data for selected country
    country_df = df[df["country"] == country].sort_values("year").copy()

    # Features and predictions for all years
    feature_cols = [c for c in df.columns if c not in ["country", "countryiso3code", "year", "gdp_growth_next"]]
    X_country = country_df[feature_cols]
    country_df["predicted_gdp_growth"] = model.predict(X_country)

    # Show latest available prediction
    st.subheader(f"📊 Predictions for {country}")
    st.write(country_df[["year", "gdp_growth", "predicted_gdp_growth"]].tail(10))

    # Plot actual vs predicted
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(country_df["year"], country_df["gdp_growth"], marker="o", label="Actual GDP Growth")
    ax.plot(country_df["year"], country_df["predicted_gdp_growth"], marker="x", linestyle="--", color="red", label="Predicted GDP Growth")

    ax.set_title(f"{country} GDP Growth: Actual vs Predicted")
    ax.set_xlabel("Year")
    ax.set_ylabel("GDP Growth (%)")
    ax.legend()
    st.pyplot(fig)


if __name__ == "__main__":
    main()

