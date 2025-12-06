import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Historical data (March 17 - May 1, 2022)
historical_dates = pd.date_range(start='2022-03-17', end='2022-05-01')
historical_returnees = [1667, 4360, 4360, 4360, 1878, 2116, 2087, 2098, 2270, 2270, 2270, 
                        2342, 2331, 2550, 2729, 2619, 2619, 2619, 2528, 2743, 2432, 2717, 
                        2718, 2718, 2718, 2926, 3022, 2875, 3120, 3093, 3093, 3093, 2819, 
                        3480, 3325, 2476, 2475, 2475, 2475, 2475, 2485, 3104, 3415, 3523, 
                        3523, 3522]

# Conflict events (March 2 - August 1)
conflict_events = [136, 164, 126, 102, 87, 131, 154, 107, 114, 151, 83, 109, 153, 136, 93, 
                   121, 99, 74, 73, 65, 84, 83, 136, 111, 117, 86, 78, 130, 104, 174, 190, 
                   91, 81, 68, 110, 106, 165, 122, 98, 104, 98, 76, 59, 114, 110, 84, 108, 
                   97, 101, 62, 121, 118, 128, 83, 122, 95, 140, 162, 114, 122, 120, 108, 
                   112, 110, 138, 111, 88, 108, 109, 103, 89, 166, 155, 113, 145, 139, 134, 
                   118, 178, 117, 124, 156, 105, 134, 123, 110, 132, 129, 134, 173, 133, 
                   119, 145, 154, 146, 127, 161, 136, 149, 133, 124, 148, 147, 124, 117, 
                   147, 143, 109, 149, 139, 208, 155, 171, 197, 113, 133, 196, 123, 206, 
                   103, 222, 190, 227, 152, 120, 161, 154, 226, 161, 167, 178, 209, 214, 
                   256, 269, 170, 209, 196, 195, 163, 215, 266, 211, 170, 204, 197, 179, 
                   226, 285, 209, 220, 241, 192]

fatalities = [277, 296, 140, 157, 121, 156, 212, 93, 83, 118, 54, 140, 265, 1198, 374, 107, 
              149, 96, 51, 473, 52, 43, 228, 241, 44, 20, 37, 43, 11, 208, 88, 2, 93, 62, 
              36, 20, 69, 95, 22, 126, 36, 106, 5, 69, 81, 44, 21, 84, 22, 37, 318, 99, 40, 
              144, 73, 88, 11, 111, 23, 47, 160, 12, 68, 16, 76, 38, 171, 26, 27, 180, 978, 
              112, 335, 19, 242, 51, 207, 20, 206, 72, 10, 82, 10, 65, 20, 102, 78, 162, 
              156, 78, 18, 60, 160, 11, 118, 177, 127, 109, 21, 184, 12, 49, 31, 18, 82, 
              14, 105, 1, 66, 10, 24, 35, 25, 142, 1, 58, 22, 48, 41, 13, 52, 29, 14, 47, 
              17, 14, 9, 37, 2, 333, 51, 134, 36, 107, 273, 13, 26, 122, 132, 83, 82, 150, 
              0, 88, 126, 32, 1, 234, 192, 114, 92, 46, 8]

# Generate forecast dates (March 24 - August 1)
forecast_dates = pd.date_range(start='2022-03-24', end='2022-08-01')
start_date = datetime(2022, 3, 2)

# Model parameters
baseline = 2750  # Average from stable period (Mar 21 - May 1)
decay_half_life = 45  # days
conflict_sensitivity = -3.5  # returnees decrease per conflict event above baseline
fatality_sensitivity = -0.8  # returnees decrease per fatality above baseline
weekend_effect = 0.92  # 8% reduction on weekends

# Calculate baseline conflict/fatality (first 50 days average)
baseline_conflict = np.mean(conflict_events[:50])
baseline_fatality = np.mean(fatalities[:50])

# Generate forecasts
forecasts = []

for date in forecast_dates:
    days_since_start = (date - start_date).days
    day_index = days_since_start  # Index in conflict arrays
    
    # Time decay component (exponential)
    days_since_march_21 = (date - datetime(2022, 3, 21)).days
    decay_factor = 0.5 ** (days_since_march_21 / decay_half_life)
    
    # Conflict intensity effect
    if day_index < len(conflict_events):
        current_conflict = conflict_events[day_index]
        current_fatality = fatalities[day_index]
    else:
        current_conflict = baseline_conflict
        current_fatality = baseline_fatality
    
    conflict_impact = conflict_sensitivity * (current_conflict - baseline_conflict) / 10
    fatality_impact = fatality_sensitivity * (current_fatality - baseline_fatality) / 10
    
    # Weekend effect (Saturday = 5, Sunday = 6)
    weekday_factor = weekend_effect if date.weekday() >= 5 else 1.0
    
    # Combined forecast
    forecast = baseline * decay_factor * weekday_factor + conflict_impact + fatality_impact
    
    # Ensure non-negative and add some random variation
    forecast = max(500, forecast)  # Minimum 500 returnees/day
    
    # Special adjustments for known patterns
    if date.month == 7 and date.day >= 15:  # Late July - conflict intensifies
        forecast *= 0.85
    
    if date.month == 8:  # August - very low returns
        forecast *= 0.80
    
    forecasts.append(int(round(forecast)))

# Create DataFrame
df = pd.DataFrame({
    'time': forecast_dates.strftime('%Y-%m-%d'),
    'estimated_returnee': forecasts
})

# Save to CSV
csv_output = df.to_csv('llm_datasets/output/full_context/claude/Romania/run-3.csv',index=False)
#print(csv_output)

# Summary statistics
print(f"\n--- Forecast Summary ---")
print(f"Date range: {forecast_dates[0].strftime('%Y-%m-%d')} to {forecast_dates[-1].strftime('%Y-%m-%d')}")
print(f"Total days: {len(forecasts)}")
print(f"Average daily returnees: {np.mean(forecasts):.0f}")
print(f"Range: {min(forecasts)} - {max(forecasts)}")
print(f"Total estimated returnees: {sum(forecasts):,}")