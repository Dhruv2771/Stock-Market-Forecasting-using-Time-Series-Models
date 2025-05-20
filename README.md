Stock Market Forecasting Using Time Series Models
Description
This project applies time series models to forecast stock market closing prices. It includes data collection from Yahoo Finance and implementation of ARIMA, Prophet, and LSTM models. The performance of each model is evaluated using RMSE, and visualizations are provided for model insights.
Project Structure
stock-market-forecasting/
├── data/                      # For storing raw or processed data
├── models/                    # Saved models if applicable
├── notebooks/                 # Jupyter notebooks (optional)
├── scripts/                   # Python scripts (e.g., arima.py, lstm.py)
├── README.md                  # Project overview and documentation
├── requirements.txt           # Python dependencies
├── main.py                    # Main pipeline to run the forecasting
├── LICENSE                    # Project license (e.g., MIT)
Models Implemented
- ARIMA – Autoregressive Integrated Moving Average model
- Prophet – Facebook’s additive time series forecasting model
- LSTM – Long Short-Term Memory deep learning model using TensorFlow/Keras
Installation
```bash
# Clone the repo
git clone https://github.com/your-username/stock-market-forecasting.git
cd stock-market-forecasting

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
Usage
```bash
python main.py
```
This script will:
- Download historical stock data (default: AAPL)
- Plot the closing price
- Fit ARIMA, Prophet, and LSTM models
- Evaluate and visualize forecasts with RMSE
Example Output
Each model generates:
- Forecast plot
- RMSE score for last 100 days of prediction
Requirements (requirements.txt)
yfinance
matplotlib
pandas
numpy
scikit-learn
statsmodels
prophet
tensorflow
License
This project is licensed under the MIT License – see the LICENSE file for details.
