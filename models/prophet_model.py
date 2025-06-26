import pandas as pd
from prophet import Prophet

def train_prophet_model(csv_path):
    df = pd.read_csv(csv_path)
    df['data'] = pd.to_datetime(df['data'])
    df = df.rename(columns={'data': 'ds', 'venda': 'y'})

    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=30)  # Forecast for next 30 days
    forecast = model.predict(future)
    
    return df, forecast, model




