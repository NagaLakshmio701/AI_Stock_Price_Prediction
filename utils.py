import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

def predict_stock(ticker,days):
    # Load model
    model_path = "model\Stock Predictions Model.keras"

    model = load_model(model_path)

    # Load your data (this should be the same data used before saving the model)
    df = pd.read_csv(r"data\Apple_Stock_Data.csv")  # Make sure this file exists with Close column
    df = df.sort_values("Date")
    close_prices = df['Close'].values.reshape(-1, 1)

    # Scaling
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_prices)

    # Sequence length (should match what you used when training)
    seq_length = 60
    current_batch = scaled_data[-seq_length:].reshape((1, seq_length, 1))

    future_predictions = []
    for _ in range(days):
        next_pred = model.predict(current_batch, verbose=0)[0]
        future_predictions.append(next_pred)
        current_batch = np.append(current_batch[:, 1:, :], [[next_pred]], axis=1)

    # Inverse scale
    future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # Generate dates
    start_date = pd.to_datetime(df['Date'].max())
    prediction_dates = pd.bdate_range(start=start_date + pd.Timedelta(days=1), periods=days)

    return pd.DataFrame({
        'Date': prediction_dates,
        'Predicted Price': future_prices.flatten()
    })
