# train_all_models.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
import joblib
import os

print("--- Starting Multi-Model AI Training ---")
df = pd.read_csv('ganga_multi_parameter_data.csv')

# List of parameters we want to create a model for
# In train_all_models.py
PARAMETERS_TO_TRAIN = [
    'rainfall_mm',
    'water_level_meters', # Add this new line
    'flow_m3_s', 'temperature_celsius', 'do_mg_L', 'bod_mg_L', 
    'nitrate_mg_L', 'fecal_coliform_mpn_100ml'
]

MODELS_DIR = 'models' # We'll save all models in a 'models' folder
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# Loop through each parameter and train a dedicated model
for param in PARAMETERS_TO_TRAIN:
    print("\n" + "="*50)
    print(f"  TRAINING MODEL FOR: {param}")
    print("="*50)

    series = df[param].values
    X, y = [], []
    n_past, n_future = 10, 3

    for i in range(len(series) - n_past - n_future + 1):
        X.append(series[i:i + n_past])
        y.append(series[i + n_past:i + n_past + n_future])
    X, y = np.array(X), np.array(y)
    print(f"Created {X.shape[0]} training examples.")

    print(f"Training XGBoost model for {param}...")
    base_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=300, random_state=42)
    multi_output_model = MultiOutputRegressor(estimator=base_model)
    multi_output_model.fit(X, y)
    print("✅ Training complete.")

    # Save the trained model with a specific name
    model_filename = f"{param}_model.pkl"
    model_path = os.path.join(MODELS_DIR, model_filename)
    joblib.dump(multi_output_model, model_path)
    print(f"✅ Model saved to: {model_path}")


print("\n--- ALL MODELS TRAINED SUCCESSFULLY ---")