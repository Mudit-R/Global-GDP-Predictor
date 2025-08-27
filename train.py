"""
train.py
Advanced GDP growth prediction training with multiple models and ensemble methods.
"""

import os
import joblib
import numpy as np
import pandas as pd
import warnings
from math import sqrt
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from fetch_data import fetch_worldbank, build_features, get_feature_importance_groups

warnings.filterwarnings('ignore')

ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (handles zero values)."""
    return 100 * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    )

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics."""
    return {
        "rmse": sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "mape": np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    }

def time_group_cv_score(model, X: pd.DataFrame, y: pd.Series, groups: pd.Series, 
                       scaler=None) -> Dict[str, float]:
    """Cross-validation with GroupKFold for different model types."""
    rmses, maes, smapes, r2s = [], [], [], []
    
    for train_idx, test_idx in GroupKFold(n_splits=5).split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        if scaler:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled, X_test_scaled = X_train, X_test
        
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = calculate_metrics(y_test.values, preds)
        rmses.append(metrics["rmse"])
        maes.append(metrics["mae"])
        smapes.append(metrics["smape"])
        r2s.append(metrics["r2"])
    
    return {
        "rmse": np.mean(rmses),
        "mae": np.mean(maes),
        "smape": np.mean(smapes),
        "r2": np.mean(r2s),
        "rmse_std": np.std(rmses),
        "mae_std": np.std(maes),
        "smape_std": np.std(smapes),
        "r2_std": np.std(r2s)
    }

def train_models(X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> Dict:
    """Train multiple models and return their performance metrics."""
    print("🚀 Training multiple models...")
    
    models = {}
    results = {}
    
    # 1. Random Forest (Baseline)
    print("🌳 Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_metrics = time_group_cv_score(rf, X, y, groups)
    results["RandomForest"] = rf_metrics
    models["RandomForest"] = rf
    
    # 2. XGBoost
    print("📈 Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_metrics = time_group_cv_score(xgb_model, X, y, groups)
    results["XGBoost"] = xgb_metrics
    models["XGBoost"] = xgb_model
    
    # 3. Gradient Boosting
    print("🔥 Training Gradient Boosting...")
    gb = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    gb_metrics = time_group_cv_score(gb, X, y, groups)
    results["GradientBoosting"] = gb_metrics
    models["GradientBoosting"] = gb
    
    # 4. Ridge Regression
    print("🏔️  Training Ridge Regression...")
    scaler = StandardScaler()
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge_metrics = time_group_cv_score(ridge, X, y, groups, scaler)
    results["Ridge"] = ridge_metrics
    models["Ridge"] = ridge
    
    # 5. SVR
    print("🔍 Training Support Vector Regression...")
    svr = SVR(kernel='rbf', C=100, gamma='scale')
    svr_metrics = time_group_cv_score(svr, X, y, groups, scaler)
    results["SVR"] = svr_metrics
    models["SVR"] = svr
    
    # 6. Ensemble (Voting Regressor)
    print("🎯 Creating Ensemble Model...")
    ensemble_models = [
        ('rf', RandomForestRegressor(n_estimators=200, random_state=42)),
        ('xgb', xgb.XGBRegressor(n_estimators=200, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=200, random_state=42))
    ]
    
    ensemble = VotingRegressor(estimators=ensemble_models, n_jobs=-1)
    ensemble_metrics = time_group_cv_score(ensemble, X, y, groups)
    results["Ensemble"] = ensemble_metrics
    models["Ensemble"] = ensemble
    
    return models, results

def plot_model_comparison(results: Dict, save_path: str):
    """Create and save model comparison visualizations."""
    # Prepare data for plotting
    metrics_df = pd.DataFrame(results).T
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # RMSE comparison
    axes[0, 0].bar(metrics_df.index, metrics_df['rmse'], color='skyblue', alpha=0.7)
    axes[0, 0].set_title('RMSE Comparison (Lower is Better)')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # MAE comparison
    axes[0, 1].bar(metrics_df.index, metrics_df['mae'], color='lightgreen', alpha=0.7)
    axes[0, 1].set_title('MAE Comparison (Lower is Better)')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # SMAPE comparison
    axes[1, 0].bar(metrics_df.index, metrics_df['smape'], color='lightcoral', alpha=0.7)
    axes[1, 0].set_title('SMAPE Comparison (Lower is Better)')
    axes[1, 0].set_ylabel('SMAPE (%)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # R² comparison
    axes[1, 1].bar(metrics_df.index, metrics_df['r2'], color='gold', alpha=0.7)
    axes[1, 1].set_title('R² Comparison (Higher is Better)')
    axes[1, 1].set_ylabel('R²')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Model comparison plot saved to: {save_path}")

def save_models_and_metrics(models: Dict, results: Dict, X: pd.DataFrame, y: pd.Series):
    """Save all trained models and their performance metrics."""
    print("💾 Saving models and metrics...")
    
    # Save individual models
    for name, model in models.items():
        joblib.dump(model, os.path.join(ARTIFACTS_DIR, f"{name.lower()}_model.pkl"))
    
    # Save metrics
    metrics_df = pd.DataFrame(results).T
    metrics_df.to_csv(os.path.join(ARTIFACTS_DIR, "all_models_metrics.csv"))
    
    # Save feature names
    feature_names = pd.DataFrame({
        'feature_name': X.columns.tolist(),
        'feature_index': range(len(X.columns))
    })
    feature_names.to_csv(os.path.join(ARTIFACTS_DIR, "feature_names.csv"), index=False)
    
    # Save best model info
    best_model = metrics_df['rmse'].idxmin()
    best_metrics = metrics_df.loc[best_model]
    
    best_model_info = {
        'best_model': best_model,
        'best_rmse': best_metrics['rmse'],
        'best_mae': best_metrics['mae'],
        'best_smape': best_metrics['smape'],
        'best_r2': best_metrics['r2']
    }
    
    pd.DataFrame([best_model_info]).to_csv(os.path.join(ARTIFACTS_DIR, "best_model_info.csv"), index=False)
    
    print(f"🏆 Best model: {best_model} (RMSE: {best_metrics['rmse']:.4f})")
    
    return best_model

def main():
    """Main training pipeline."""
    print("🌍 Global GDP Predictor - Advanced Training Pipeline")
    print("=" * 60)
    
    # 1. Prepare dataset
    print("\n📊 Step 1: Preparing dataset...")
    df = fetch_worldbank()
    df = build_features(df)
    
    # Features and target
    feature_cols = [c for c in df.columns if c not in ["country", "countryiso3code", "year", "gdp_growth_next"]]
    X = df[feature_cols]
    y = df["gdp_growth_next"]
    groups = df["country"]
    
    print(f"✅ Dataset prepared: {len(df)} samples, {len(feature_cols)} features")
    print(f"🌍 Countries: {len(df['country'].unique())}")
    print(f"📅 Years: {df['year'].min()} - {df['year'].max()}")
    
    # 2. Train multiple models
    print("\n🤖 Step 2: Training multiple models...")
    models, results = train_models(X, y, groups)
    
    # 3. Display results
    print("\n📈 Model Performance Results:")
    print("-" * 80)
    metrics_df = pd.DataFrame(results).T
    print(metrics_df.round(4))
    
    # 4. Create visualizations
    print("\n🎨 Step 3: Creating visualizations...")
    plot_model_comparison(results, os.path.join(ARTIFACTS_DIR, "model_comparison.png"))
    
    # 5. Save everything
    print("\n💾 Step 4: Saving models and artifacts...")
    best_model_name = save_models_and_metrics(models, results, X, y)
    
    # 6. Final summary
    print("\n🎉 Training completed successfully!")
    print(f"🏆 Best performing model: {best_model_name}")
    print(f"📁 All artifacts saved in: {os.path.abspath(ARTIFACTS_DIR)}")
    
    # Display feature importance for best model
    if best_model_name in ["RandomForest", "XGBoost", "GradientBoosting"]:
        print(f"\n🔍 Feature importance for {best_model_name}:")
        best_model = models[best_model_name]
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(feature_importance.head(10))

if __name__ == "__main__":
    main()
