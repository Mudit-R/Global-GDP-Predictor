"""
fetch_data.py

Enhanced World Bank API fetch with more indicators and advanced feature engineering.
Works reliably on Python 3.13+.
"""
import pandas as pd
import requests
import datetime
import numpy as np
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# Enhanced World Bank indicator codes
DEFAULT_INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",              # GDP growth (annual %)
    "FP.CPI.TOTL.ZG": "cpi_annual_pct",             # Inflation (CPI, %)
    "FS.AST.PRVT.GD.ZS": "domestic_credit_pct_gdp", # Domestic credit to private sector (% GDP)
    "NV.IND.TOTL.ZS": "industry_pct_gdp",           # Industry share of GDP
    "NV.AGR.TOTL.ZS": "agri_pct_gdp",               # Agriculture share of GDP
    "NE.CON.PRVT.PC.KD": "private_consumption_pc",  # Private consumption per capita
    "NE.EXP.GNFS.ZS": "exports_pct_gdp",            # Exports of goods and services (% GDP)
    "NE.IMP.GNFS.ZS": "imports_pct_gdp",            # Imports of goods and services (% GDP)
    "GC.DOD.TOTL.GD.ZS": "government_debt_pct_gdp", # Government debt (% GDP)
    "NY.GDP.PCAP.KD.ZG": "gdp_per_capita_growth",   # GDP per capita growth (%)
    "SL.UEM.TOTL.ZS": "unemployment_rate",          # Unemployment rate (%)
    "FR.INR.RINR": "real_interest_rate",            # Real interest rate (%)
    "NE.RSB.GNFS.ZS": "trade_balance_pct_gdp",      # Trade balance (% GDP)
    "SE.XPD.TOTL.GD.ZS": "education_expenditure",   # Education expenditure (% GDP)
    "SH.MED.BEDS.ZS": "hospital_beds_per_1000",     # Hospital beds per 1000 people
    "EN.ATM.CO2E.PC": "co2_emissions_per_capita",   # CO2 emissions per capita
    "IT.NET.USER.ZS": "internet_users_pct"          # Internet users (% of population)
}

def fetch_worldbank(indicators: dict = None, countries: List[str] = None,
                    start_year: int = 1990, end_year: int = None) -> pd.DataFrame:
    """
    Fetch World Bank data with enhanced error handling and retry logic.
    """
    if indicators is None:
        indicators = DEFAULT_INDICATORS
    if end_year is None:
        end_year = datetime.datetime.now().year

    all_data = []
    
    print(f"🌍 Fetching data for {len(indicators)} indicators from {start_year} to {end_year}...")

    for code, name in indicators.items():
        url = f"https://api.worldbank.org/v2/country/all/indicator/{code}?date={start_year}:{end_year}&format=json&per_page=20000"
        print(f"📊 Fetching {name} ({code})...")
        
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                print(f"⚠️  Warning: Failed to fetch {code}: {resp.status_code}")
                continue
                
            data = resp.json()
            if len(data) < 2 or not data[1]:
                print(f"⚠️  Warning: No data for {code}")
                continue
                
            for row in data[1]:
                if row.get("value") is not None:
                    all_data.append({
                        "country": row["country"]["value"],
                        "countryiso3code": row["countryiso3code"],
                        "year": int(row["date"]),
                        name: float(row["value"])
                    })
                    
        except Exception as e:
            print(f"❌ Error fetching {code}: {str(e)}")
            continue

    if not all_data:
        raise RuntimeError("No data could be fetched from World Bank API")

    df = pd.DataFrame(all_data)
    print(f"✅ Successfully fetched {len(df)} data points")

    # Get the actual columns that were successfully fetched
    actual_columns = [col for col in df.columns if col not in ["country", "countryiso3code", "year"]]
    
    # Pivot into wide format (one row per country-year)
    df = df.pivot_table(index=["country", "countryiso3code", "year"],
                        values=actual_columns,
                        aggfunc="first").reset_index()
    
    # Clean up country names
    df = df[df["country"] != "World"]
    
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Advanced feature engineering with lag/rolling features, interactions, and transformations.
    """
    df = df.copy().sort_values(["country", "year"]).reset_index(drop=True)
    
    print("🔧 Building advanced features...")

    # Target: next year's GDP growth
    df["gdp_growth_next"] = df.groupby("country")["gdp_growth"].shift(-1)
    
    # Remove rows without target
    df = df.dropna(subset=["gdp_growth_next"])
    
    # Economic ratios and derived features
    if all(col in df.columns for col in ["exports_pct_gdp", "imports_pct_gdp"]):
        df["trade_openness"] = df["exports_pct_gdp"] + df["imports_pct_gdp"]
        df["trade_balance"] = df["exports_pct_gdp"] - df["imports_pct_gdp"]
    
    if all(col in df.columns for col in ["industry_pct_gdp", "agri_pct_gdp"]):
        df["industrialization_ratio"] = df["industry_pct_gdp"] / (df["agri_pct_gdp"] + 1e-8)
    
    # Credit and financial features
    if "domestic_credit_pct_gdp" in df.columns:
        df["credit_yoy"] = df.groupby("country")["domestic_credit_pct_gdp"].pct_change()
        df["credit_acceleration"] = df.groupby("country")["credit_yoy"].pct_change()
    
    # Inflation and monetary features
    if "cpi_annual_pct" in df.columns:
        df["inflation_volatility"] = df.groupby("country")["cpi_annual_pct"].rolling(3).std().reset_index(0, drop=True)
    
    # Lag and rolling features for all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ["year", "gdp_growth_next"]]
    
    for col in numeric_cols:
        if col in df.columns and df[col].notna().sum() > 0:
            # Lag features
            df[f"{col}_lag1"] = df.groupby("country")[col].shift(1)
            df[f"{col}_lag2"] = df.groupby("country")[col].shift(2)
            
            # Rolling features
            df[f"{col}_roll3_mean"] = (
                df.groupby("country")[col]
                  .rolling(window=3, min_periods=1).mean()
                  .reset_index(0, drop=True)
            )
            df[f"{col}_roll3_std"] = (
                df.groupby("country")[col]
                  .rolling(window=3, min_periods=1).std()
                  .reset_index(0, drop=True)
            )
            df[f"{col}_roll5_mean"] = (
                df.groupby("country")[col]
                  .rolling(window=5, min_periods=1).mean()
                  .reset_index(0, drop=True)
            )
    
    # Interaction features
    if all(col in df.columns for col in ["cpi_annual_pct", "gdp_growth"]):
        df["inflation_gdp_interaction"] = df["cpi_annual_pct"] * df["gdp_growth"]
    
    if all(col in df.columns for col in ["domestic_credit_pct_gdp", "gdp_growth"]):
        df["credit_gdp_interaction"] = df["domestic_credit_pct_gdp"] * df["gdp_growth"]
    
    # Year-over-year changes for key indicators
    key_indicators = ["gdp_growth", "cpi_annual_pct", "domestic_credit_pct_gdp", "unemployment_rate"]
    for col in key_indicators:
        if col in df.columns:
            df[f"{col}_yoy_change"] = df.groupby("country")[col].pct_change()
    
    # Global economic cycle features (average across countries)
    global_features = ["gdp_growth", "cpi_annual_pct", "domestic_credit_pct_gdp"]
    for col in global_features:
        if col in df.columns:
            global_avg = df.groupby("year")[col].mean()
            df[f"global_{col}_avg"] = df["year"].map(global_avg)
            df[f"{col}_vs_global"] = df[col] - df[f"global_{col}_avg"]
    
    print(f"✅ Built {len(df.columns)} features from {len(numeric_cols)} base indicators")
    
    # Final cleanup - remove columns with too many missing values
    threshold = len(df) * 0.5  # Remove columns with more than 50% missing values
    df = df.dropna(axis=1, thresh=threshold)
    
    # Clean infinite values and extreme outliers
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col in df.columns:
            # Replace infinite values with NaN
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
            # Remove extreme outliers (beyond 3 standard deviations)
            if df[col].notna().sum() > 0:
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:
                    lower_bound = mean_val - 3 * std_val
                    upper_bound = mean_val + 3 * std_val
                    df.loc[df[col] < lower_bound, col] = np.nan
                    df.loc[df[col] > upper_bound, col] = np.nan
    
    # Handle missing values with imputation
    print("🔧 Handling missing values...")
    
    # For each numeric column, fill missing values with appropriate methods
    for col in numeric_columns:
        if col in df.columns and df[col].isna().sum() > 0:
            if col == "gdp_growth_next":
                # Don't impute the target variable - drop those rows
                continue
            elif col in ["gdp_growth", "cpi_annual_pct", "unemployment_rate"]:
                # For key economic indicators, use forward fill then backward fill
                df[col] = df.groupby("country")[col].fillna(method='ffill').fillna(method='bfill')
            else:
                # For other features, use median imputation
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
    
    # Final cleanup - remove rows with missing target variable
    df = df.dropna(subset=["gdp_growth_next"])
    
    print(f"✅ Data cleaning completed. Final dataset: {len(df)} samples, {len(df.columns)} features")
    
    return df

def get_feature_importance_groups() -> Dict[str, List[str]]:
    """
    Group features by category for better analysis and visualization.
    """
    return {
        "Macroeconomic": ["gdp_growth", "cpi_annual_pct"],
        "Financial": ["domestic_credit_pct_gdp", "credit_yoy", "credit_acceleration", "real_interest_rate"],
        "Structural": ["industry_pct_gdp", "agri_pct_gdp", "industrialization_ratio"],
        "Trade": ["exports_pct_gdp", "imports_pct_gdp", "trade_openness", "trade_balance"],
        "Social": ["unemployment_rate", "education_expenditure", "hospital_beds_per_1000"],
        "Technology": ["internet_users_pct"],
        "Government": ["government_debt_pct_gdp"],
        "Consumption": ["private_consumption_pc", "gdp_per_capita_growth"]
    }
