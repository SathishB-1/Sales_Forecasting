## Sales Forecasting Project:

This project provides a modular implementation of three time-series forecasting models—Prophet, ARIMA, and LSTM—to predict future sales based on historical data.

## Features:

📤 Upload your own CSV file (max 200MB)

📆 Select columns for date and sales

📊 Forecast next 30 days

📈 Visualize actual vs forecasted values

⚙️ Choose between Prophet, ARIMA, or LSTM

## Project Structure

sales_forecaste/
│
├── data/
│   └── cleaned_mock_kaggle.csv     
│
├── models/
│   ├── prophet_model.py              
│   ├── arima_model.py                  
│   └── lstm_model.py                   
│
├── visualize/
│   └── forecast_plot.py                
│
├── requirements.txt                    
└── README.md                           


## Installation Clone the repository:

git clone https://github.com/yourusername/sales_forecaste.git

cd sales_forecaste

## Create a virtual environment (optional but recommended):

python -m venv venv

source venv/bin/activate  # on Windows: venv\Scripts\activate

## Install dependencies:

pip install -r requirements.txt

## Run the App:

streamlit run app.py

## Models Used:

Model	          Description

Prophet	          Handles trend + seasonality, robust for business time-series.

ARIMA	          Classical time-series model (AutoRegressive Integrated Moving Average).

LSTM	          Deep Learning model trained on past sequences to predict future.

## TODO / Future Improvements:

 Let user define custom forecast horizon (e.g. 7, 14, 60 days)

 Download forecast result as CSV

 Show model performance metrics (MAE, RMSE)

 Add training visualizations (loss curve for LSTM)

## Forecasting Models:

 Facebook Prophet, Statsmodels, TensorFlow Keras

## License:

This project is licensed under the MIT License. See LICENSE for details.


## Benefits of the Project:

1. Multi-Model Comparison:

🔄 Compare traditional (ARIMA), statistical (Prophet), and deep learning (LSTM) approaches.

Helps determine the best fit for different types of sales data (linear, seasonal, or complex trends).

2. Custom Dataset Support:

📤 Accepts user-defined datasets.

Works with real-world business data across various industries like retail, manufacturing, and e-commerce.

3. Modular Design:

🧱 Each model is implemented in its own module (prophet_model.py, arima_model.py, lstm_model.py).

Easily extendable—add new models or improve existing ones without breaking the structure.

4. Ease of Use:

🎯 Simple interface (main.py) runs all models and plots the forecasts.

Minimal configuration needed—just supply your dataset with a date and sales column.

5. Visual Forecast Insights:

📊 Clear visualization of forecast vs. actual data.

Helps in understanding model performance and confidence visually before using metrics.

Future Prediction:

![image alt](https://github.com/SathishB-1/Sales_Forecasting/blob/f068a5610d1a27c2d9a3de0fb263b422783edbc0/Screenshot%202025-06-24%20214707.png)
