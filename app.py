import datetime

import numpy as np
import torch
import torch.nn as nn
import yfinance as yf
from flask import Flask, jsonify, render_template, request
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)


# Define the same LSTM model class used during training
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take the last output
        out = self.fc(out)
        return out


@app.route("/")
def index():
    widget_stats = {"symbol": "AMZN", "interval": "D"}
    return render_template("index.html", widget_stats=widget_stats)


@app.route("/evaluate", methods=["POST"])
def evaluate():
    data = request.get_json()
    symbol = data.get("symbol")
    date_str = data.get("date")

    # Validate date format
    try:
        chosen_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    except Exception:
        return jsonify({"error": "Invalid date format"}), 400

    # Download historical data (300 days before chosen date)
    start_date = chosen_date - datetime.timedelta(days=300)
    end_date = chosen_date.strftime("%Y-%m-%d")
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date.strftime("%Y-%m-%d"), end=end_date)

    if df.empty or df.shape[0] < 60:
        return jsonify({"error": "Not enough historical data or stock not found"}), 400

    # Prepare last 60 days of Close and EMA for LSTM input
    seq_length = 60
    df["EMA"] = df["Close"].ewm(span=20, adjust=False).mean()

    features = ["Close", "EMA"]
    data_features = df[features].tail(seq_length).to_numpy()

    # Normalize with MinMaxScaler (fit on current data)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_features)

    input_tensor = (
        torch.FloatTensor(data_scaled).unsqueeze(0).to("cpu")
    )  # shape: (1, 60, 2)

    # Load trained model
    input_size = 2
    hidden_size = 128
    num_layers = 3
    dropout = 0.01

    try:
        model = LSTMModel(input_size, hidden_size, num_layers, dropout).to("cpu")
        model.load_state_dict(
            torch.load("model/best_model.pth", map_location=torch.device("cpu"))
        )
    except Exception as e:
        print(f"Model load/init error: {e}")
        return jsonify({"error": "Model loading failed"}), 500

    model.eval()

    # Predict and inverse transform
    try:
        with torch.no_grad():
            pred_norm = model(input_tensor).item()

        dummy_row = np.zeros((1, len(features)))
        dummy_row[0, features.index("Close")] = pred_norm

        pred_inverse = scaler.inverse_transform(dummy_row)[0, features.index("Close")]
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed"}), 500

    last_close = df["Close"].iloc[-1]
    decision = "Buy" if pred_inverse > last_close else "Sell"

    response = {
        "predicted_price": round(pred_inverse, 2),
        "decision": decision,
    }

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
