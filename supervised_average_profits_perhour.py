import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

MODEL_PATH = "taxi_profit_prediction_model.keras"

# ----------------------------
# Load or define model
# ----------------------------
if os.path.exists(MODEL_PATH):
    print(f"[INFO] Found existing model at '{MODEL_PATH}'. Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    model = None  # Will train later if needed

# ----------------------------
# Prediction function
# ----------------------------
def predict_avg_profit(day_hour_str):
    """
    Predict average profit using a trained model based on input like 'Mon 05'.
    """
    if model is None:
        print("[ERROR] Model not found and must be trained first.")
        return

    try:
        day_part, hour_part = day_hour_str.strip().split()
        hour = int(hour_part)
        if hour < 0 or hour > 23:
            raise ValueError("Hour must be between 0 and 23")
    except Exception as e:
        print(f"[ERROR] Invalid format. Use format like 'Mon 05'. Reason: {e}")
        return

    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    if day_part not in day_labels:
        print(f"[ERROR] Day must be one of: {', '.join(day_labels)}")
        return

    day_index = day_labels.index(day_part)
    day_norm = day_index / 6.0
    hour_norm = hour / 23.0
    input_array = np.array([[day_norm, hour_norm]], dtype=np.float32)
    predicted_profit = model.predict(input_array, verbose=0).flatten()[0]
    print(f"[RESULT] Predicted average profit for {day_part} {hour:02d}: ${predicted_profit:.2f}")

# ----------------------------
# If called from terminal with a time argument, just predict and exit
# ----------------------------
if __name__ == "__main__" and len(sys.argv) == 2:
    predict_avg_profit(sys.argv[1])
    sys.exit()

# ----------------------------
# Training / Plotting Mode
# Only runs if no prediction argument
# ----------------------------
print("[INFO] Loading and preparing data...")

df = pd.read_csv("cleaned_taxi_data_with_tip_percent.csv")
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
df['pickup_day'] = df['day_of_week']

day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
               'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
df['pickup_day'] = df['pickup_day'].map(day_mapping)

grouped = df.groupby(['pickup_day', 'pickup_hour'])['total_amount'].mean().reset_index(name='avg_profit')
print("[INFO] Grouped and averaged data complete.")

grouped['pickup_day_norm'] = grouped['pickup_day'] / 6.0
grouped['pickup_hour_norm'] = grouped['pickup_hour'] / 23.0

x = grouped[['pickup_day_norm', 'pickup_hour_norm']]
y = grouped['avg_profit']

print("[INFO] Splitting data...")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)
print(f"[INFO] Training size: {len(x_train)} | Test size: {len(x_test)}")

# ----------------------------
# Build and train model if not found earlier
# ----------------------------
if model is None:
    print("[INFO] No saved model found. Building and training new model...")
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(2,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='mae', patience=20, restore_best_weights=True)

    history = model.fit(
        x_train, y_train,
        epochs=100,
        batch_size=1,
        validation_data=(x_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )
    print("[INFO] Training complete. Saving model...")
    model.save(MODEL_PATH)

# ----------------------------
# Evaluate and visualize results
# ----------------------------
print("[INFO] Evaluating model...")
loss, mae = model.evaluate(x_test, y_test, verbose=1)
print(f"[RESULT] Test MAE: ${mae:.2f}")

y_pred = model.predict(x_test).flatten()

x_test_denorm = x_test.copy()
x_test_denorm.columns = ['pickup_hour_norm', 'pickup_day_norm']
x_test_denorm['pickup_day'] = (x_test_denorm['pickup_day_norm'] * 6).round().astype(int)
x_test_denorm['pickup_hour'] = (x_test_denorm['pickup_hour_norm'] * 23).round().astype(int)
x_test_denorm['time_slot'] = x_test_denorm['pickup_day'] * 24 + x_test_denorm['pickup_hour']

day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
x_test_denorm['label'] = x_test_denorm.apply(
    lambda row: f"{day_labels[int(row['pickup_day'])]} {int(row['pickup_hour']):02d}", axis=1
)

plt.figure(figsize=(20, 6))
plt.scatter(x_test_denorm['time_slot'], y_test, color='blue', label='Actual Profit', alpha=0.6, marker='o')
plt.scatter(x_test_denorm['time_slot'], y_pred, color='orange', label='Predicted Profit', alpha=0.6, marker='x')

sorted_labels = x_test_denorm[['time_slot', 'label']].drop_duplicates().sort_values('time_slot')
tick_positions = sorted_labels['time_slot'].values
tick_labels = sorted_labels['label'].values

plt.xticks(tick_positions, tick_labels, rotation=75, fontsize=8)
plt.title("Actual vs Predicted Average Profit Per Ride by Hour")
plt.xlabel("Day and Hour")
plt.ylabel("Avg Profit ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
