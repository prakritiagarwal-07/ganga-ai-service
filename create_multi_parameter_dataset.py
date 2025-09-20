# create_multi_parameter_dataset.py (Final Calibrated Version)
import pandas as pd
import numpy as np

print("--- Generating final, calibrated, multi-parameter historical dataset ---")
n_days = 365 * 5
dates = pd.to_datetime([pd.to_datetime('2020-01-01') + pd.DateOffset(days=i) for i in range(n_days)])

# --- Calibrated Simulation for Each Parameter ---

# 1. Rainfall (mm)
rainfall = np.random.uniform(0, 1, n_days) * 5
for i, date in enumerate(dates):
    if 6 <= date.month <= 9:  # Monsoon season
        rainfall[i] += np.random.uniform(0, 1) ** 2 * 40 # More realistic heavy bursts
rainfall = np.clip(rainfall, 0, 150) # Clip at a max of 150mm/day

# 2. Water Level (meters) - Controlled and stable
water_level = np.zeros(n_days)
base_level = 70.0  # Normal level for Varanasi
water_level[0] = base_level
for i in range(1, n_days):
    rainfall_effect = (rainfall[i] / 50) + (rainfall[i-1] / 25)
    natural_runoff = 0.18 # Increased runoff to prevent constant growth
    new_level = water_level[i-1] + rainfall_effect - natural_runoff + np.random.normal(0, 0.05)
    water_level[i] = np.clip(new_level, 68.5, 74.0) # Tighter, realistic range

# 3. Flow (cubic m/s) - Directly linked to water level
flow = 800 + (water_level - base_level) * 150 + np.random.normal(0, 50)
flow = np.clip(flow, 300, 3000)

# 4. Temperature (Celsius) - Smooth yearly cycle
temperature = 25 - 9 * np.cos((pd.Series(dates).dt.dayofyear - 30) * 2 * np.pi / 365) + np.random.normal(0, 0.4, n_days)
temperature = np.clip(temperature, 15, 33)

# 5. Pollutants - Calibrated to realistic ranges
base_bod, base_nitrate, base_fecal = 5.5, 4.5, 7500
bod = base_bod + (flow / 1200) + np.random.normal(0, 0.4, n_days) + (np.random.rand(n_days) < 0.02) * 4
bod = np.clip(bod, 2, 15)

nitrate = base_nitrate + (flow / 1000) + np.random.normal(0, 0.3, n_days)
nitrate = np.clip(nitrate, 2, 18)

fecal_coliform = np.zeros(n_days)
fecal_coliform[0] = base_fecal
for i in range(1, n_days):
    runoff_effect = rainfall[i] * 100 + rainfall[i-1] * 50
    spike = np.random.uniform(4000, 7000) if np.random.rand() < 0.015 else 0
    natural_decay = fecal_coliform[i-1] * 0.1 # Bacteria die off
    fecal_coliform[i] = fecal_coliform[i-1] + runoff_effect + spike - natural_decay + np.random.normal(0, 400)
fecal_coliform = np.clip(fecal_coliform, 1000, 25000)

# 6. Dissolved Oxygen (DO) (mg/L) - Linked to all pollutants and temperature
base_do = 9.8
do = base_do - (bod / 3.5) - (nitrate / 4.5) - ((temperature - 15) / 10) + np.random.normal(0, 0.2, n_days)
do = np.clip(do, 3.5, 9.5)

# --- Create and Save the Final, High-Quality DataFrame ---
df = pd.DataFrame({
    'date': dates,
    'rainfall_mm': rainfall.round(2),
    'water_level_meters': water_level.round(2),
    'flow_m3_s': flow.round(2),
    'temperature_celsius': temperature.round(2),
    'do_mg_L': do.round(2),
    'bod_mg_L': bod.round(2),
    'nitrate_mg_L': nitrate.round(2),
    'fecal_coliform_mpn_100ml': fecal_coliform.round(0)
})
df.to_csv('ganga_multi_parameter_data.csv', index=False)
print("✅ Successfully created final, calibrated 'ganga_multi_parameter_data.csv'.")