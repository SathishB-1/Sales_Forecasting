import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def train_arima_model(csv_path):
    df = pd.read_csv(csv_path)
    df['data'] = pd.to_datetime(df['data'])
    df.set_index('data', inplace=True)

    model = ARIMA(df['venda'], order=(5,1,0))  # example order, tune it
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=30)
    return df, forecast
