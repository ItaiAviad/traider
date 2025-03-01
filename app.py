from flask import Flask, render_template, request, jsonify
import datetime
import yfinance as yf
import torch

from model import DQN, preprocess_obs

app = Flask(__name__)


@app.route("/")
def index():
    # Pass default widget stats to the template.
    widget_stats = {"symbol": "GOOG", "interval": "D"}
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

    today = datetime.date.today()
    if chosen_date > today:
        return jsonify({"error": "Date is in the future"}), 400

    # Download historical data (300-day buffer)
    start_date = chosen_date - datetime.timedelta(days=300)
    end_date = chosen_date.strftime("%Y-%m-%d")
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date.strftime("%Y-%m-%d"), end=end_date)

    if df.empty or df.shape[0] < 200:
        return jsonify({"error": "Not enough historical data or stock not found"}), 400

    # Use the last 200 days; take the last 10 days as the observation window.
    data_200 = df.tail(200)
    window_size = 10  # must match training setup
    obs_data = data_200["Close"].tail(window_size).to_numpy()
    observation = obs_data.reshape(window_size, 1)

    # Setup the model
    input_dim = window_size  # flattened observation of shape (window_size,)
    output_dim = 2  # 2 actions: Buy and Sell
    device = torch.device("cpu")
    model = DQN(input_dim, output_dim).to(device)

    try:
        model.load_state_dict(torch.load("dqn_trading_model.pth", map_location=device))
    except Exception:
        return jsonify({"error": "Model weights not found"}), 500
    model.eval()

    obs_flat = preprocess_obs(observation)
    obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = model(obs_tensor)
        action = q_values.argmax().item()

    decision = "Sell" if action == 1 else "Buy"
    return jsonify({"result": decision})


if __name__ == "__main__":
    app.run(debug=True)
