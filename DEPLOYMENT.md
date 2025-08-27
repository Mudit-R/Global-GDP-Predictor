# 🚀 Deployment Guide - Global GDP Predictor Pro

This guide covers multiple deployment options for the Global GDP Predictor Pro application.

## 📋 Prerequisites

- Python 3.10+ or Docker
- Git
- Internet connection (for World Bank API access)

## 🐳 Option 1: Docker Deployment (Recommended)

### Quick Start with Docker Compose

```bash
# Clone the repository
git clone <your-repo-url>
cd Global-GDP-Predictor

# Build and run with Docker Compose
docker-compose up --build

# Access the application
open http://localhost:8501
```

### Manual Docker Build

```bash
# Build the Docker image
docker build -t global-gdp-predictor .

# Run the container
docker run -p 8501:8501 -v $(pwd)/artifacts:/app/artifacts global-gdp-predictor

# Access the application
open http://localhost:8501
```

## 🐍 Option 2: Local Python Environment

### 1. Setup Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Models

```bash
# Train all models (this may take 10-30 minutes)
python train.py
```

### 3. Launch Dashboard

```bash
# Start the Streamlit application
streamlit run app.py
```

## ☁️ Option 3: Cloud Deployment

### Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy automatically

### Heroku

```bash
# Install Heroku CLI
# Create Procfile
echo "web: streamlit run app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

### Google Cloud Run

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/gdp-predictor
gcloud run deploy gdp-predictor --image gcr.io/YOUR_PROJECT_ID/gdp-predictor --platform managed
```

## 🔧 Configuration

### Environment Variables

```bash
# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true

# API Configuration
WORLDBANK_API_TIMEOUT=30
WORLDBANK_API_RETRIES=3

# Model Configuration
MODEL_CACHE_SIZE=1000
PREDICTION_CONFIDENCE_LEVEL=0.95
```

### Custom Configuration File

Create `config.yaml`:

```yaml
app:
  title: "Global GDP Predictor Pro"
  theme: "light"
  layout: "wide"

models:
  default: "Ensemble"
  cache_size: 1000
  confidence_level: 0.95

api:
  worldbank:
    timeout: 30
    retries: 3
    indicators:
      - "NY.GDP.MKTP.KD.ZG"
      - "FP.CPI.TOTL.ZG"
      - "FS.AST.PRVT.GD.ZS"

features:
  lag_features: true
  rolling_features: true
  interaction_features: true
  global_features: true
```

## 📊 Performance Optimization

### Model Caching

```python
# Enable model caching for faster predictions
@st.cache_resource
def load_models():
    # Models are cached in memory
    return load_all_models()
```

### Data Preprocessing

```python
# Use efficient data types
df = df.astype({
    'year': 'int16',
    'gdp_growth': 'float32',
    'country': 'category'
})
```

### Batch Predictions

```python
# Process multiple countries at once
def batch_predict(model, countries_data):
    predictions = model.predict(countries_data)
    return predictions
```

## 🔒 Security Considerations

### API Rate Limiting

```python
import time
from functools import wraps

def rate_limit(calls_per_second=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            time.sleep(1/calls_per_second)
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

### Input Validation

```python
def validate_country_input(country: str) -> bool:
    valid_countries = get_valid_countries()
    return country in valid_countries

def validate_year_range(start_year: int, end_year: int) -> bool:
    return 1990 <= start_year <= end_year <= 2024
```

## 📈 Monitoring and Logging

### Application Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

### Performance Monitoring

```python
import time

def monitor_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logging.info(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper
```

## 🚨 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   ```bash
   # Ensure models are trained first
   python train.py
   ```

2. **API Connection Issues**
   ```bash
   # Check internet connection
   curl https://api.worldbank.org/v2/country/all/indicator/NY.GDP.MKTP.KD.ZG
   ```

3. **Memory Issues**
   ```bash
   # Reduce batch size or use data streaming
   export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
   ```

4. **Port Conflicts**
   ```bash
   # Use different port
   streamlit run app.py --server.port=8502
   ```

### Debug Mode

```bash
# Enable debug logging
export STREAMLIT_LOG_LEVEL=debug
streamlit run app.py
```

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [World Bank API Documentation](https://datahelpdesk.worldbank.org/knowledgebase/articles/889386-developer-information-overview)
- [Docker Documentation](https://docs.docker.com/)
- [Plotly Documentation](https://plotly.com/python/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Need Help?** Open an issue on GitHub or contact the development team.
