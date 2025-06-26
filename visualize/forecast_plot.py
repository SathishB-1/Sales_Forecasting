import matplotlib.pyplot as plt

def plot_forecast(df, forecast, method='prophet'):
    plt.figure(figsize=(12, 6))
    
    if method == 'prophet':
        plt.plot(df['ds'], df['y'], label='Actual Sales')
        plt.plot(forecast['ds'], forecast['yhat'], label='Forecast (Prophet)')
    else:
        plt.plot(df.index, df['venda'], label='Actual Sales')
        future_dates = pd.date_range(start=df.index[-1], periods=len(forecast)+1, freq='D')[1:]
        plt.plot(future_dates, forecast, label=f'Forecast ({method.upper()})')

    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title('Sales Forecast vs Actual')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()