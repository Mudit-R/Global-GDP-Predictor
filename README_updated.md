# 🌍 Global GDP Predictor

A machine learning project that predicts **next-year GDP growth** for any country using macroeconomic indicators from the **World Bank API**.  
Includes a **Streamlit dashboard** for interactive exploration and model predictions.

---

## 🚀 Features
- 📊 Fetches World Bank indicators (GDP growth, inflation, credit, industry, agriculture, consumption, etc.).
- 🏗️ Feature engineering with **lagged** and **rolling** variables.
- 🤖 Trains and evaluates **Random Forest** (and extendable to XGBoost).
- 📈 Time-aware cross-validation with **GroupKFold** (country-wise splits).
- 🔍 Explainability with **SHAP values** (feature importance per prediction).
- 🎛️ Interactive **Streamlit dashboard** for:
  - Country-wise GDP growth predictions
  - Comparison of actual vs predicted growth
  - Visualizations of macro indicators

---

## 📂 Project Structure
```
Global-GDP-Predictor/
│
├── app.py            # Streamlit dashboard
├── fetch_data.py     # Fetch & preprocess World Bank data
├── train.py          # Model training & artifact saving
├── utils.py          # Helper functions (plotting, SHAP, model loading)
├── requirements.txt  # Python dependencies
└── README.md         # Project documentation
```

Artifacts (trained models, metrics) are saved in the `artifacts/` folder after training.

---

## ⚡ Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/Global-GDP-Predictor.git
cd Global-GDP-Predictor
```

### 2. Setup environment
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Train models (optional)
This step fetches World Bank data and trains a Random Forest model.
```bash
python train.py
```

### 4. Launch dashboard
```bash
streamlit run app.py
```

---

## 📊 Example
Once launched, the Streamlit app allows you to:
- Select a country from the dropdown
- View actual vs predicted GDP growth for recent years
- Plot time series of indicators (inflation, credit growth, etc.)
- Inspect prediction drivers using SHAP

---

## 🛠️ Tech Stack
- **Python 3.10+**
- **scikit-learn** (Random Forest)
- **XGBoost** (optional)
- **Streamlit** (interactive dashboard)
- **SHAP** (model explainability)
- **World Bank API** (macroeconomic data)

---

## 🔮 Future Work
- Add more models (Prophet, LSTM for time-series)
- Integrate forecast uncertainty
- Build country-level dashboards with richer visualization (Plotly, Altair)
- Deploy on Streamlit Cloud / Hugging Face Spaces

---

## 👨‍💻 Author
Developed by *[Your Name]* — Data Science & Machine Learning Enthusiast.  
Feel free to connect on [LinkedIn](https://linkedin.com/in/yourprofile) or check out my [GitHub](https://github.com/your-username).
