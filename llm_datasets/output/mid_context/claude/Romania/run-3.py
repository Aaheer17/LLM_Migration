import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Data from context
outward_flow = [189196, 191870, 201993, 176462, 165995, 164690, 142435, 152534, 129194, 120893, 141482, 113838, 107187, 130498, 108001, 116562, 100351, 104209, 82411, 67261, 67083, 69999, 53350, 74871, 70110, 68476, 56113, 49014, 68558, 52681, 58902, 59703, 52775, 40788, 28951, 42533, 40726, 53012, 58207, 42250, 40309, 35879, 30843, 18361, 36847, 35580, 46170, 22900, 46612, 22319, 16089, 29490, 28601, 34560, 17109, 43464, 26695, 29461, 34737, 31843, 22860, 22070, 24839, 31791, 33778, 18843, 27369, 16417, 15557, 18191, 17205, 16255, 19923, 21301, 15151, 20568, 16103, 23320, 17094, 16356, 20602, 17226, 19547, 10816, 16834, 16806, 15920, 19457, 9131, 14516, 16003, 43554, 21041, 11771, 9861, 13283, 11695, 12413, 15627, 8969, 9624, 14877, 12272, 10640, 8096, 24245, 12522, 9227, 12048, 15587, 7149, 9222, 6448, 12299, 15013, 4791, 22843, 15903, 14904, 11073, 7343, 7704, 16360, 15469, 8341, 4698, 11647, 5134, 12084, 8047, 6080, 5662, 10422, 9022, 12002, 20498, 12746, 18657, 4691, 8384, 7276, 11640, 10485, 8505, 10656, 9563, 9219, 11864, 8148, 11166, 13076, 4289, 8441]

conflict_events = [136, 164, 126, 102, 87, 131, 154, 107, 114, 151, 83, 109, 153, 136, 93, 121, 99, 74, 73, 65, 84, 83, 136, 111, 117, 86, 78, 130, 104, 174, 190, 91, 81, 68, 110, 106, 165, 122, 98, 104, 98, 76, 59, 114, 110, 84, 108, 97, 101, 62, 121, 118, 128, 83, 122, 95, 140, 162, 114, 122, 120, 108, 112, 110, 138, 111, 88, 108, 109, 103, 89, 166, 155, 113, 145, 139, 134, 118, 178, 117, 124, 156, 105, 134, 123, 110, 132, 129, 134, 173, 133, 119, 145, 154, 146, 127, 161, 136, 149, 133, 124, 148, 147, 124, 117, 147, 143, 109, 149, 139, 208, 155, 171, 197, 113, 133, 196, 123, 206, 103, 222, 190, 227, 152, 120, 161, 154, 226, 161, 167, 178, 209, 214, 256, 269, 170, 209, 196, 195, 163, 215, 266, 211, 170, 204, 197, 179, 226, 285, 209, 220, 241, 192]

fatalities = [277, 296, 140, 157, 121, 156, 212, 93, 83, 118, 54, 140, 265, 1198, 374, 107, 149, 96, 51, 473, 52, 43, 228, 241, 44, 20, 37, 43, 11, 208, 88, 2, 93, 62, 36, 20, 69, 95, 22, 126, 36, 106, 5, 69, 81, 44, 21, 84, 22, 37, 318, 99, 40, 144, 73, 88, 11, 111, 23, 47, 160, 12, 68, 16, 76, 38, 171, 26, 27, 180, 978, 112, 335, 19, 242, 51, 207, 20, 206, 72, 10, 82, 10, 65, 20, 102, 78, 162, 156, 78, 18, 60, 160, 11, 118, 177, 127, 109, 21, 184, 12, 49, 31, 18, 82, 14, 105, 1, 66, 10, 24, 35, 25, 142, 1, 58, 22, 48, 41, 13, 52, 29, 14, 47, 17, 14, 9, 37, 2, 333, 51, 134, 36, 107, 273, 13, 26, 122, 132, 83, 82, 150, 0, 88, 126, 32, 1, 234, 192, 114, 92, 46, 8]

# Start date for data
start_date = datetime(2022, 3, 2)
forecast_start = datetime(2022, 3, 24)
forecast_end = datetime(2022, 8, 1)

# Calculate day indices
forecast_start_idx = (forecast_start - start_date).days
forecast_end_idx = (forecast_end - start_date).days + 1

# Generate forecast
dates = []
returnees = []

for day_idx in range(forecast_start_idx, forecast_end_idx):
    current_date = start_date + timedelta(days=day_idx)
    dates.append(current_date.strftime('%Y-%m-%d'))
    
    # Days since invasion (Feb 24, 2022)
    days_since_invasion = (current_date - datetime(2022, 2, 24)).days
    
    # Base return proportion (increases logarithmically over time)
    # Starts at ~2% and grows to ~10% by August
    base_return_rate = 0.02 + 0.025 * np.log(days_since_invasion / 10)
    base_return_rate = min(base_return_rate, 0.12)  # Cap at 12%
    
    # Lag period: use outward flow from 10 days ago
    lag_days = 10
    lagged_idx = day_idx - lag_days
    if lagged_idx >= 0 and lagged_idx < len(outward_flow):
        lagged_outflow = outward_flow[lagged_idx]
    else:
        lagged_outflow = 50000  # Default estimate
    
    # Conflict damping factor
    if day_idx < len(fatalities):
        daily_fatalities = fatalities[day_idx]
        # High fatalities (>150) suppress returns significantly
        if daily_fatalities > 200:
            conflict_damping = 0.5
        elif daily_fatalities > 100:
            conflict_damping = 0.7
        else:
            conflict_damping = 0.85 + 0.15 * (1 - daily_fatalities/100)
            conflict_damping = min(conflict_damping, 1.0)
    else:
        conflict_damping = 0.85
    
    # Romania share factor (estimated 15-20% of total returnees use Romania route)
    romania_share = 0.18
    
    # Weekly pattern (lower on weekends)
    day_of_week = current_date.weekday()
    if day_of_week >= 5:  # Weekend
        weekly_factor = 0.7
    else:
        weekly_factor = 1.0
    
    # Seasonal boost (summer months - people returning for agriculture, etc.)
    month = current_date.month
    if month in [6, 7, 8]:
        seasonal_boost = 1.2
    else:
        seasonal_boost = 1.0
    
    # Calculate returnees
    estimated_returnees = (
        lagged_outflow * 
        base_return_rate * 
        conflict_damping * 
        romania_share * 
        weekly_factor * 
        seasonal_boost
    )
    
    # Add some randomness to simulate real-world variation
    noise = np.random.normal(1.0, 0.15)
    estimated_returnees = int(estimated_returnees * noise)
    
    # Ensure non-negative
    estimated_returnees = max(0, estimated_returnees)
    
    returnees.append(estimated_returnees)

# Create DataFrame
df = pd.DataFrame({
    'time': dates,
    'estimated_returnee': returnees
})

# Display first 20 rows and statistics
print("Return Migration Forecast: Romania to Ukraine")
print("=" * 60)
print(f"\nForecast Period: {dates[0]} to {dates[-1]}")
print(f"Total Days: {len(dates)}")
print(f"\nStatistics:")
print(f"Mean Daily Returnees: {np.mean(returnees):.0f}")
print(f"Median Daily Returnees: {np.median(returnees):.0f}")
print(f"Min Daily Returnees: {np.min(returnees):.0f}")
print(f"Max Daily Returnees: {np.max(returnees):.0f}")
print(f"Total Estimated Returnees: {np.sum(returnees):.0f}")
print("\n" + "=" * 60)
print("\nFirst 20 rows:")
print(df.head(20).to_string(index=False))
print("\n...")
print("\nLast 10 rows:")
print(df.tail(10).to_string(index=False))

# Save to CSV format
print("\n" + "=" * 60)
print("\nComplete CSV Output:")
print("=" * 60)
print(df.to_csv('run-3.csv',index=False))
