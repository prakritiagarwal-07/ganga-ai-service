# ai_server.py - The Complete Python AI "Brain" Service
import os
from flask import Flask, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# --- PART 1: LOAD ALL AI MODELS AND DATA AT STARTUP ---
MODELS_DIR = 'models'
DATA_FILE = 'ganga_multi_parameter_data.csv'
models = {}
df_history = None
try:
    df_history = pd.read_csv(DATA_FILE)
    for filename in os.listdir(MODELS_DIR):
        if filename.endswith(".pkl"):
            param_name = filename.replace("_model.pkl", "")
            models[param_name] = joblib.load(os.path.join(MODELS_DIR, filename))
    print(f"✅ All {len(models)} AI models and the dataset were loaded successfully.")
except Exception as e:
    print(f"❌ A critical error occurred during initialization: {e}")
    models = None

# --- PART 2: THE ALERTING AND INFERENCE LOGIC ---
THRESHOLDS = {
    "water_level_meters": {"limit": 72.5, "condition": "above", "name": "Ganga Water Level"},
    "do_mg_L": {"limit": 5.0, "condition": "below", "name": "Dissolved Oxygen (DO)"},
    "bod_mg_L": {"limit": 8.0, "condition": "above", "name": "Biochemical Oxygen Demand (BOD)"},
    "fecal_coliform_mpn_100ml": {"limit": 2500, "condition": "above", "name": "Fecal Coliform"}
}
def check_for_alerts(param, forecast_values):
    if param not in THRESHOLDS: return None
    rule = THRESHOLDS[param]
    if param == "water_level_meters" and max(forecast_values) > rule["limit"]:
        return f"🚨 FLOOD ALERT: {rule['name']} is forecast to reach {max(forecast_values)}m, exceeding the danger level of {rule['limit']}m."
    if rule["condition"] == "above" and max(forecast_values) > rule["limit"]:
        return f"⚠️ High Level Warning: {rule['name']} is forecast to reach {max(forecast_values)}."
    elif rule["condition"] == "below" and min(forecast_values) < rule["limit"]:
        return f"⚠️ Low Level Warning: {rule['name']} is forecast to drop to {min(forecast_values)}."
    return None

def infer_pollution_source(forecasts):
    if max(forecasts.get('bod_mg_L', [0])) > 8.0 and max(forecasts.get('fecal_coliform_mpn_100ml', [0])) > 20000:
        return "Untreated sewage outflow is the likely primary source."
    if max(forecasts.get('nitrate_mg_L', [0])) > 10.0:
        return "Agricultural runoff is a likely contributing source."
    return "Pollution levels appear to be within standard limits."

# --- PART 3: THE MAIN API ENDPOINT ---
@app.route("/api/predict/varanasi", methods=['GET'])
def predict_varanasi():
    if not models:
        return jsonify({"error": "AI models are not available."}), 500

    all_forecasts = {}
    all_alerts = []

    for param, model in models.items():
        input_data = df_history[param].tail(10).tolist()
        features = np.array(input_data).reshape(1, -1)
        forecast = model.predict(features)[0]
        all_forecasts[param] = [round(float(p), 2) for p in forecast]
        alert_message = check_for_alerts(param, all_forecasts[param])
        if alert_message:
            all_alerts.append(alert_message)

    inferred_source = infer_pollution_source(all_forecasts)

    return jsonify({
        "location": "Varanasi",
        "forecasts": all_forecasts,
        "alerts": all_alerts,
        "source_inference": inferred_source
    })

# --- PART 4: START THE SERVER ---
if __name__ == "__main__":
    print("--- Starting AI Forecasting & Alerting Server ---")
    app.run(port=5000, debug=True)