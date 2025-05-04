import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Load the trained model
MODEL_PATH = "taxi_profit_prediction_model.keras"
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model file not found at: {MODEL_PATH}")
    exit(1)

model = tf.keras.models.load_model(MODEL_PATH)

# Define days and hours
day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
results = []

# Generate predictions for every day-hour combo
for day_index, day in enumerate(day_labels):
    for hour in range(24):
        day_norm = day_index / 6.0
        hour_norm = hour / 23.0
        input_data = np.array([[day_norm, hour_norm]], dtype=np.float32)
        prediction = model.predict(input_data, verbose=0).flatten()[0]
        results.append({
            "day": day,
            "day_index": day_index,
            "hour": hour,
            "predicted_profit": prediction
        })

# Convert to DataFrame and save to CSV
df = pd.DataFrame(results)
df.to_csv("predicted_profit_by_day_hour.csv", index=False)
print("[INFO] Saved predictions to predicted_profit_by_day_hour.csv")

# Plotting
plt.figure(figsize=(12, 6))
for day in day_labels:
    daily_data = df[df["day"] == day]
    plt.plot(daily_data["hour"], daily_data["predicted_profit"], label=day)

plt.title("Predicted Average Profit by Hour for Each Day")
plt.xlabel("Hour of Day")
plt.ylabel("Predicted Average Profit ($)")
plt.xticks(ticks=range(24), labels=[f"{h:02d}:00" for h in range(24)])  # Add this line
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
