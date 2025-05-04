import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("cleaned_taxi_data_with_tip_percent.csv")
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')

df['hour'] = df['tpep_pickup_datetime'].dt.hour

# Create pivot: rows = day, columns = hour
ride_counts = df.pivot_table(
    index='day_of_week',
    columns='hour',
    values='id',
    aggfunc='count',
    fill_value=0
)

# Order the days properly
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
ride_counts = ride_counts.reindex(day_order)

# Plot heatmap
plt.figure(figsize=(12, 6))
plt.title("Number of Rides by Day of Week and Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Day of Week")

plt.xticks(ticks=np.arange(24), labels=ride_counts.columns)
plt.yticks(ticks=np.arange(len(day_order)), labels=day_order)

plt.imshow(ride_counts, aspect='auto', cmap='viridis', interpolation='nearest')
plt.colorbar(label='Number of Rides')

plt.tight_layout()
plt.show()
