import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def train_lstm_model(csv_path):
    df = pd.read_csv(csv_path)
    df['data'] = pd.to_datetime(df['data'])
    df.set_index('data', inplace=True)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['venda']])

    def create_sequences(data, seq_length=10):
        x, y = [], []
        for i in range(len(data) - seq_length):
            x.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(x), np.array(y)

    x, y = create_sequences(scaled_data)

    x = x.reshape((x.shape[0], x.shape[1], 1))

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(x.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(x, y, epochs=20, verbose=0)

    # Forecast next 30 days
    predictions = []
    last_seq = x[-1]

    for _ in range(30):
        pred = model.predict(last_seq.reshape(1, -1, 1), verbose=0)
        predictions.append(pred[0, 0])
        last_seq = np.concatenate((last_seq[1:], [[pred[0, 0]]]), axis=0)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return df, predictions.flatten()
