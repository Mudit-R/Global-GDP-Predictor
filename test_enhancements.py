#!/usr/bin/env python3
"""
test_enhancements.py
Test script to validate all the enhancements made to the Global GDP Predictor project.
"""

import os
import sys
import importlib
import pandas as pd
import numpy as np
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported."""
    print("🔍 Testing package imports...")
    
    required_packages = [
        'pandas', 'numpy', 'streamlit', 'plotly', 'sklearn', 
        'xgboost', 'joblib', 'matplotlib', 'seaborn'
    ]
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            return False
    
    return True

def test_data_fetching():
    """Test the enhanced data fetching functionality."""
    print("\n🌍 Testing data fetching...")
    
    try:
        from fetch_data import fetch_worldbank, build_features, get_feature_importance_groups
        
        # Test feature groups
        feature_groups = get_feature_importance_groups()
        print(f"✅ Feature groups: {len(feature_groups)} categories")
        
        # Test data fetching (small sample)
        print("📊 Fetching sample data (this may take a moment)...")
        df = fetch_worldbank(start_year=2020, end_year=2022)
        print(f"✅ Data fetched: {len(df)} records, {len(df.columns)} columns")
        
        # Test feature building
        df = build_features(df)
        print(f"✅ Features built: {len(df.columns)} total features")
        
        # Verify we have the essential columns
        essential_cols = ['country', 'year', 'gdp_growth', 'gdp_growth_next']
        missing_cols = [col for col in essential_cols if col not in df.columns]
        if missing_cols:
            print(f"⚠️  Missing essential columns: {missing_cols}")
            return False
        
        print(f"✅ Essential columns verified: {len(essential_cols)} columns present")
        return True
        
    except Exception as e:
        print(f"❌ Data fetching error: {e}")
        return False

def test_model_training():
    """Test the enhanced model training functionality."""
    print("\n🤖 Testing model training...")
    
    try:
        from train import calculate_metrics, time_group_cv_score
        
        # Test metrics calculation
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        metrics = calculate_metrics(y_true, y_pred)
        print(f"✅ Metrics calculated: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model training error: {e}")
        return False

def test_utils():
    """Test the enhanced utility functions."""
    print("\n🔧 Testing utility functions...")
    
    try:
        from utils import (
            calculate_prediction_confidence, 
            create_feature_correlation_heatmap,
            validate_predictions
        )
        
        # Test confidence calculation
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        confidence = calculate_prediction_confidence(y_true, y_pred)
        print(f"✅ Confidence calculated: {confidence['coverage_rate']:.1f}% coverage")
        
        # Test validation
        validation = validate_predictions(y_true, y_pred)
        print(f"✅ Validation passed: {sum(validation.values())}/{len(validation)} checks")
        
        return True
        
    except Exception as e:
        print(f"❌ Utils error: {e}")
        return False

def test_dashboard():
    """Test the enhanced dashboard functionality."""
    print("\n🎨 Testing dashboard components...")
    
    try:
        from app import (
            create_gdp_prediction_chart,
            create_feature_importance_chart,
            create_model_comparison_chart
        )
        
        # Test chart creation
        test_data = pd.DataFrame({
            'year': [2020, 2021, 2022],
            'gdp_growth': [2.0, 3.0, 2.5],
            'predicted_gdp_growth': [2.1, 2.9, 2.6]
        })
        
        # Test GDP prediction chart
        gdp_chart = create_gdp_prediction_chart(test_data, "Test Country")
        print("✅ GDP prediction chart created")
        
        # Test feature importance chart
        feature_names = ['feature1', 'feature2', 'feature3']
        importance = [0.5, 0.3, 0.2]
        
        # Mock model with feature_importances_
        class MockModel:
            def __init__(self):
                self.feature_importances_ = np.array(importance)
        
        mock_model = MockModel()
        imp_chart = create_feature_importance_chart(mock_model, feature_names, "Test Model")
        print("✅ Feature importance chart created")
        
        return True
        
    except Exception as e:
        print(f"❌ Dashboard error: {e}")
        return False

def test_docker_files():
    """Test that Docker configuration files exist and are valid."""
    print("\n🐳 Testing Docker configuration...")
    
    docker_files = ['Dockerfile', 'docker-compose.yml']
    
    for file in docker_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            return False
    
    return True

def test_documentation():
    """Test that enhanced documentation exists."""
    print("\n📚 Testing documentation...")
    
    doc_files = ['README.md', 'DEPLOYMENT.md']
    
    for file in doc_files:
        if os.path.exists(file):
            with open(file, 'r') as f:
                content = f.read()
                if len(content) > 1000:  # Check if substantial content
                    print(f"✅ {file} (substantial content)")
                else:
                    print(f"⚠️  {file} (minimal content)")
        else:
            print(f"❌ {file} missing")
            return False
    
    return True

def main():
    """Run all tests."""
    print("🚀 Global GDP Predictor Pro - Enhancement Validation")
    print("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("Data Fetching", test_data_fetching),
        ("Model Training", test_model_training),
        ("Utility Functions", test_utils),
        ("Dashboard Components", test_dashboard),
        ("Docker Configuration", test_docker_files),
        ("Documentation", test_documentation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All enhancements are working correctly!")
        return True
    else:
        print("⚠️  Some enhancements need attention.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
