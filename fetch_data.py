"""
fetch_data.py

Direct World Bank API fetch (no wbdata, no cache issues).
Works reliably on Python 3.13+.
"""
import pandas as pd
import requests
import datetime
from typing import List

# World Bank indicator codes
DEFAULT_INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",              # GDP growth (annual %)
    "FP.CPI.TOTL.ZG": "cpi_annual_pct",             # Inflation (CPI, %)
    "FS.AST.PRVT.GD.ZS": "domestic_credit_pct_gdp", # Domestic credit to private sector (% GDP)
    "NV.IND.TOTL.ZS": "industry_pct_gdp",           # Industry share of GDP
    "NV.AGR.TOTL.ZS": "agri_pct_gdp",               # Agriculture share of GDP
    "NE.CON.PRVT.PC.KD": "private_consumption_pc"   # Private consumption per capita
}

def fetch_worldbank(indicators: dict = None, countries: List[str] = None,
                    start_year: int = 1990, end_year: int = None) -> pd.DataFrame:
    if indicators is None:
        indicators = DEFAULT_INDICATORS
    if end_year is None:
        end_year = datetime.datetime.now().year

    all_data = []

    for code, name in indicators.items():
        url = f"https://api.worldbank.org/v2/country/all/indicator/{code}?date={start_year}:{end_year}&format=json&per_page=20000"
        print(f"Fetching {name} ({code})...")
        resp = requests.get(url)
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to fetch {code}: {resp.text}")
        data = resp.json()[1]  # second element has the data
        for row in data:
            all_data.append({
                "country": row["country"]["value"],
                "countryiso3code": row["countryiso3code"],
                "year": int(row["date"]),
                name: row["value"]
            })

    df = pd.DataFrame(all_data)

    # Pivot into wide format (one row per country-year)
    df = df.pivot_table(index=["country", "countryiso3code", "year"],
                        values=list(indicators.values()),
                        aggfunc="first").reset_index()
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer lag/rolling features and target variable."""
    df = df.copy().sort_values(["country", "year"]).reset_index(drop=True)

    # Target: next year's GDP growth
    df["gdp_growth_next"] = df.groupby("country")["gdp_growth"].shift(-1)

    # YoY credit growth proxy
    if "domestic_credit_pct_gdp" in df.columns:
        df["credit_yoy"] = df.groupby("country")["domestic_credit_pct_gdp"].pct_change()

    # Lag and rolling features
    lag_cols = ["cpi_annual_pct", "industry_pct_gdp", "agri_pct_gdp",
                "private_consumption_pc", "credit_yoy"]
    for col in lag_cols:
        if col in df.columns:
            df[f"{col}_lag1"] = df.groupby("country")[col].shift(1)
            df[f"{col}_lag2"] = df.groupby("country")[col].shift(2)
            df[f"{col}_roll3"] = (
                df.groupby("country")[col]
                  .rolling(window=3, min_periods=1).mean()
                  .reset_index(0, drop=True)
            )

    return df.dropna(subset=["gdp_growth_next"])
