# run_prediction_and_alerts.py (Final Version with Flood Alert)
import pandas as pd
import numpy as np
import joblib
import os

print("\n--- AI Prediction & Alerting System ---")

# --- PART 1: THE ALERTING SYSTEM "RULEBOOK" (UPGRADED) ---
THRESHOLDS = {
    # NEW: Flood Alert Rule
    "water_level_meters": {
        "limit": 72.5, # Danger level for Varanasi is ~72.5 meters
        "condition": "above",
        "name": "Ganga Water Level"
    },
    "do_mg_L": {"limit": 5.0, "condition": "below", "name": "Dissolved Oxygen (DO)"},
    "bod_mg_L": {"limit": 8.0, "condition": "above", "name": "Biochemical Oxygen Demand (BOD)"},
    "nitrate_mg_L": {"limit": 10.0, "condition": "above", "name": "Nitrate"},
    "fecal_coliform_mpn_100ml": {"limit": 20000, "condition": "below", "name": "Fecal Coliform"},
    "temperature_celsius": {"limit": 30.0, "condition": "above", "name": "Water Temperature"}
}

# --- The rest of the script remains the same ---
def check_for_alerts(param, forecast_values):
    if param not in THRESHOLDS: return None
    rule = THRESHOLDS[param]
    # Custom message for the critical flood alert
    if param == "water_level_meters" and max(forecast_values) > rule["limit"]:
        return f"🚨 FLOOD ALERT: {rule['name']} is forecast to reach {max(forecast_values)}m, exceeding the danger level of {rule['limit']}m."

    if rule["condition"] == "above" and max(forecast_values) > rule["limit"]:
        return f"⚠️ Low Level Warning: {rule['name']} is forecast to drop {max(forecast_values)}."
    elif rule["condition"] == "below" and min(forecast_values) < rule["limit"]:
        return f"⚠️ High Level Warning: {rule['name']} is forecast to reached to {min(forecast_values)}."
    return None

# Load models and data
MODELS_DIR = 'models'; DATA_FILE = 'ganga_multi_parameter_data.csv'
try:
    df = pd.read_csv(DATA_FILE)
    models = {}
    for filename in os.listdir(MODELS_DIR):
        if filename.endswith(".pkl"):
            param_name = filename.replace("_model.pkl", "")
            models[param_name] = joblib.load(os.path.join(MODELS_DIR, filename))
    print(f"✅ All {len(models)} AI models and dataset loaded.")
except Exception as e:
    print(f"❌ Error during initialization: {e}"); exit()

# Get forecasts and generate alerts
all_forecasts = {}; all_alerts = []
for param, model in models.items():
    input_data = df[param].tail(10).tolist()
    features = np.array(input_data).reshape(1, -1)
    forecast = model.predict(features)[0]
    all_forecasts[param] = [round(float(p), 2) for p in forecast]
    alert_message = check_for_alerts(param, all_forecasts[param])
    if alert_message: all_alerts.append(alert_message)

# Display the final report
print("\n" + "="*60); print("        🤖 COMPREHENSIVE 3-DAY FORECAST 🤖"); print("="*60)
print(f"{'Parameter':<30} | {'Day 1':>8} | {'Day 2':>8} | {'Day 3':>8}"); print("-"*60)
for param, forecast in all_forecasts.items():
    print(f"{param:<30} | {forecast[0]:>8} | {forecast[1]:>8} | {forecast[2]:>8}")
print("="*60)

print("\n" + "="*70); print("                      📢 PUBLIC SAFETY ALERTS 📢"); print("="*70)
if not all_alerts:
    print("  ✅ All parameters are forecast to be within safe levels.")
else:
    # Prioritize the flood alert by printing it first if it exists
    flood_alert = next((a for a in all_alerts if "FLOOD ALERT" in a), None)
    if flood_alert:
        print(f"  {flood_alert}")
    for alert in all_alerts:
        if "FLOOD ALERT" not in alert:
            print(f"  {alert}")
print("="*70 + "\n")