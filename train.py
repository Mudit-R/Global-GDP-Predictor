"""
train.py
Trains GDP growth prediction models using World Bank data.
"""

import os
import joblib
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupKFold
from fetch_data import fetch_worldbank, build_features

ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error (handles zero values)."""
    return 100 * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    )


def time_group_cv_score(model, X, y, groups):
    """Cross-validation with GroupKFold, returning RMSE + MAE + SMAPE."""
    rmses, maes, smapes = [], [], []
    for train_idx, test_idx in GroupKFold(n_splits=5).split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # RMSE
        mse = mean_squared_error(y_test, preds)
        rmses.append(sqrt(mse))
        # MAE
        maes.append(mean_absolute_error(y_test, preds))
        # SMAPE
        smapes.append(smape(y_test, preds))

    return {
        "rmse": sum(rmses) / len(rmses),
        "mae": sum(maes) / len(maes),
        "smape": sum(smapes) / len(smapes),
    }


def main():
    print("Preparing dataset...")
    df = fetch_worldbank()
    df = build_features(df)

    # Features and target
    feature_cols = [c for c in df.columns if c not in ["country", "countryiso3code", "year", "gdp_growth_next"]]
    X = df[feature_cols]
    y = df["gdp_growth_next"]
    groups = df["country"]

    print(f"Samples: {len(df)}, Features: {len(feature_cols)}")

    # Random Forest model
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )

    print("Cross-validating RandomForest...")
    rf_metrics = time_group_cv_score(rf, X, y, groups)
    print("RandomForest CV:", rf_metrics)

    # Fit on full data
    rf.fit(X, y)

    # Save model + metrics
    print(">>> Saving artifacts...")
    joblib.dump(rf, os.path.join(ARTIFACTS_DIR, "rf_model.pkl"))
    pd.DataFrame([rf_metrics]).to_csv(os.path.join(ARTIFACTS_DIR, "rf_metrics.csv"), index=False)
    print(">>> Training completed. Artifacts saved in:", os.path.abspath(ARTIFACTS_DIR))


if __name__ == "__main__":
    main()
