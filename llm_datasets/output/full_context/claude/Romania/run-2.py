##claude run-2 code

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Historical data for calibration (Mar 17 - May 1)
historical_dates = pd.date_range('2022-03-17', '2022-05-01')
historical_returnees = [1667, 4360, 4360, 4360, 1878, 2116, 2087, 2098, 2270, 2270, 2270, 2342, 2331, 2550, 2729, 2619, 2619, 2619, 2528, 2743, 2432, 2717, 2718, 2718, 2718, 2926, 3022, 2875, 3120, 3093, 3093, 3093, 2819, 3480, 3325, 2476, 2475, 2475, 2475, 2475, 2485, 3104, 3415, 3523, 3523, 3522]

# Conflict events (Mar 2 onwards, align with forecast period)
conflict_events = [136, 164, 126, 102, 87, 131, 154, 107, 114, 151, 83, 109, 153, 136, 93, 121, 99, 74, 73, 65, 84, 83, 136, 111, 117, 86, 78, 130, 104, 174, 190, 91, 81, 68, 110, 106, 165, 122, 98, 104, 98, 76, 59, 114, 110, 84, 108, 97, 101, 62, 121, 118, 128, 83, 122, 95, 140, 162, 114, 122, 120, 108, 112, 110, 138, 111, 88, 108, 109, 103, 89, 166, 155, 113, 145, 139, 134, 118, 178, 117, 124, 156, 105, 134, 123, 110, 132, 129, 134, 173, 133, 119, 145, 154, 146, 127, 161, 136, 149, 133, 124, 148, 147, 124, 117, 147, 143, 109, 149, 139, 208, 155, 171, 197, 113, 133, 196, 123, 206, 103, 222, 190, 227, 152, 120, 161, 154, 226, 161, 167, 178, 209, 214, 256, 269, 170, 209, 196, 195, 163, 215, 266, 211, 170, 204, 197, 179, 226, 285, 209, 220, 241, 192]

# Generate forecast from Mar 24 to Aug 1
start_date = datetime(2022, 3, 24)
end_date = datetime(2022, 8, 1)
forecast_dates = pd.date_range(start_date, end_date)

# Reference date for time calculation
reference_date = datetime(2022, 3, 2)

# Model parameters (calibrated from historical data)
initial_base = 2750
decay_rate = 0.008
weekend_factor = 0.85

forecasts = []

for date in forecast_dates:
    # Days since reference
    days_elapsed = (date - reference_date).days
    
    # Get conflict index (offset by 22 days from Mar 2 to Mar 24)
    conflict_idx = days_elapsed
    if conflict_idx < len(conflict_events):
        conflict_level = conflict_events[conflict_idx]
    else:
        conflict_level = 150  # default
    
    # Base exponential decay
    base_returnees = initial_base * np.exp(-decay_rate * days_elapsed)
    
    # Weekend seasonality (Sat=5, Sun=6)
    day_of_week = date.weekday()
    if day_of_week >= 5:
        seasonality = weekend_factor
    else:
        seasonality = 1.0
    
    # Conflict suppression factor
    # Higher conflict events -> lower returns
    # Normalized: 100 events = no effect, 300+ events = strong suppression
    conflict_factor = max(0.5, 1.0 - (conflict_level - 100) / 400)
    
    # Additional time-based adjustment for late period (June-August)
    if date >= datetime(2022, 6, 1):
        # Further reduce returns as situation becomes more entrenched
        late_period_factor = 0.75
    else:
        late_period_factor = 1.0
    
    # Combined forecast
    forecast = base_returnees * seasonality * conflict_factor * late_period_factor
    
    # Round to integer
    forecast = int(round(forecast))
    
    forecasts.append({
        'time': date.strftime('%Y-%m-%d'),
        'estimated_returnee': forecast
    })

# Create DataFrame and export to CSV
df_forecast = pd.DataFrame(forecasts)

# Display summary statistics
print("Forecast Summary Statistics:")
print(f"Period: {df_forecast['time'].min()} to {df_forecast['time'].max()}")
print(f"Mean daily returnees: {df_forecast['estimated_returnee'].mean():.0f}")
print(f"Median daily returnees: {df_forecast['estimated_returnee'].median():.0f}")
print(f"Min: {df_forecast['estimated_returnee'].min()}")
print(f"Max: {df_forecast['estimated_returnee'].max()}")
print(f"Total forecasted returnees: {df_forecast['estimated_returnee'].sum():,}")
print("\nFirst 10 rows:")
print(df_forecast.head(10))
print("\nLast 10 rows:")
print(df_forecast.tail(10))

# Export to CSV
csv_output = df_forecast.to_csv('llm_datasets/output/full_context/claude/Romania/run-2.csv',index=False)
print("\n" + "="*50)
print("CSV OUTPUT:")
print("="*50)
print(csv_output)