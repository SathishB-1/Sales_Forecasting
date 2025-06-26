import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

from models.prophet_model import train_prophet_model
from models.arima_model import train_arima_model
from models.lstm_model import train_lstm_model
from visualize.forecast_plot import plot_forecast

csv_path = 'data/cleaned_mock_kaggle.csv'

# Prophet
df_p, forecast_p, model_p = train_prophet_model(csv_path)
plot_forecast(df_p, forecast_p, method='prophet')

# ARIMA
df_a, forecast_a = train_arima_model(csv_path)
plot_forecast(df_a, forecast_a, method='arima')

# LSTM
df_l, forecast_l = train_lstm_model(csv_path)
plot_forecast(df_l, forecast_l, method='lstm')
