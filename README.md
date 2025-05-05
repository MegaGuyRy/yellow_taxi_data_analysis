# Yellow Taxi Data Analysis

## Table of Contents
- [Overview](#overview)
- [Data Sources](#Data-Sources)
- [Tools](#Tools)
- [Data Cleaning](#Data-Cleaning)
- [Exploratiry Data Analysis](#Exploratiry-Data-Analysis)
- [Data Analysis](#Data-Analysis)
- [Results](#Results)
- [Recomendations](#Recomendations)
- [References](#References)

### Overview
Using historical yellow taxi data this data analyst project aims to find insights into most profitable times for taxi drivers to operate by analyzing fare data by hour and day of the week. The project involves data cleaning, trend analysis, and the development of a supervised machine learning model to predict average profits using historical data.

### Data Sources
The dataset I reference for this project is from NYC Open Data it contains historical data pretaining to yellow taxi cab trips. The origional datase contains 38,310,226 data entries each with 19 feature columns. The historical data ranges from 2001 to 2023 and the dataset was made public in 2024. However I reduced the dataset to features relevant to the goal of the project as follows:
-Pickup datetime
-Dropoff datetime
-Fare amount
-Tip amount
-Trip distance
-Total amount

### Tools
- Excel: initial exploration, manual cleaning, calculate and store new data.
- Python: (Pandas, NumPy, Matplotlib, TensorFlow) – data wrangling, model building, and data visualization.

### Data Cleaning
In the initial data cleaning process I removed the unnecessary features (payment_type, data_store_flag, passenger_count, ect.)as stated above using an option when I was retrieving the data set. In addition I also had to remove many of the entried becaues over 3 million entries was to much data to examine so once again I reduced the dataset now to just entries made in December 2023 the most recent data recorded in the set. Then to further enhance the ability to use this data for a machine learning model I contined to clean the set by preforming the following.
1. Added an id column to the dataset so each entry had a specific id number.
2. Dropped all of the entried where the total amount or trip distance were equal to 0 which could have been misrecoded in the dataset.
3. Added a day of the week column which would take the data from the pick_up_date column then using that data would determine what day of the week it was when that trip was recorded.
4. Added a tip percent column to more easily show the amount of the charge that was coming from the added tip.

### Exploratiry Data Analysis
EDA Reviled that the data held:
1. The least number of rides were between the early morning hours (12AM - 5AM) specificly on weekdays.
2. The highest number of rides are found specificly on fridat between the hours of (6PM - 7PM).
3. The average price per fair are almost even across each day of the week, where the highest price per fair is $28.34 is on Thursday.
4. The average price per fair are lowest on the weekends where Saturday and Sunday average $26 dollars about a $1.50 less that other days.

### Data Analysis
Before beginging to train my machine learning model I wanted to identify trends in the data that would add insights outside of predicting the most profitable times/days.

1. Initially I knew I had to measure the most profitable days on average based on the data this involved me averaging the total_amount column based on its corresponding days. I used this code below to extract and calculate the data I needed to make the bar graph.
```
# Calculate average total amount by day of week
df = pd.read_csv("cleaned_taxi_data_with_tip_percent.csv")
avg_total_amount_monday = df[df['day_of_week'] == "Monday"]['total_amount'].mean()
avg_total_amount_tuesday = df[df['day_of_week'] == "Tuesday"]['total_amount'].mean()
avg_total_amount_wednesday = df[df['day_of_week'] == "Wednesday"]['total_amount'].mean()
avg_total_amount_thursday = df[df['day_of_week'] == "Thursday"]['total_amount'].mean()
avg_total_amount_friday = df[df['day_of_week'] == "Friday"]['total_amount'].mean()
avg_total_amount_saturday = df[df['day_of_week'] == "Saturday"]['total_amount'].mean()
avg_total_amount_sunday = df[df['day_of_week'] == "Sunday"]['total_amount'].mean()

# Create bar chart for average total amount by day of week
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
averages = [
    avg_total_amount_monday,
    avg_total_amount_tuesday,
    avg_total_amount_wednesday,
    avg_total_amount_thursday,
    avg_total_amount_friday,
    avg_total_amount_saturday,
    avg_total_amount_sunday
]
```
Using this data I created the bar graph below, it shows that on average the most money made per fair is during the week as apposed to the weekend

![average_total_fair_bar](https://github.com/user-attachments/assets/183300ac-439c-4a1e-accf-8d6b04e5c135)

2. I knew that another important area to consider when trying to make maximum profits would be the peak hours on rides. To identify this I created a heat map using the python library matplotlib where I evaluated the number of rides given in a given hour for each day of the week.

![taxi_rides_heatmap](https://github.com/user-attachments/assets/cc02a2a1-ad5a-4069-ab33-ca6803b249e9)

3. I used TensorFlow to train a supervised learning model that predicts average profit based on normalized values for pickup_day and pickup_hour. The file supervised_average_profits_perhour.py can generate a new machine learing model based on the features or if there is a saved model detected it will evaluate it. I also added a function predict_avg_profit() (shown below) that you can call by entering a corresponding weekday and day combo ex("Mon 02") and it will predict the average price per fair given that day and hour combo.
```
def predict_avg_profit(day_hour_str):
    """
    Predict average profit using a trained model based on input like 'Mon 05'.
    """
    if model is None:
        print("[ERROR] Model not found and must be trained first.")
        return

    try:
        # Split input string into day and hour
        day_part, hour_part = day_hour_str.strip().split()
        hour = int(hour_part)
        # Error handling for hour
        if hour < 0 or hour > 23:
            raise ValueError("Hour must be between 0 and 23")
    except Exception as e:
        print(f"[ERROR] Invalid format. Use format like 'Mon 05'. Reason: {e}")
        return
    # Error handling for day
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    if day_part not in day_labels:
        print(f"[ERROR] Day must be one of: {', '.join(day_labels)}")
        return

    # Normalize day and hour
    day_index = day_labels.index(day_part)
    day_norm = day_index / 6.0
    hour_norm = hour / 23.0
    input_array = np.array([[day_norm, hour_norm]], dtype=np.float32)
    predicted_profit = model.predict(input_array, verbose=0).flatten()[0]
    print(f"[RESULT] Predicted average profit for {day_part} {hour:02d}: ${predicted_profit:.2f}")
```
4. Using my enginered features I created a very simple neural network using TensorFlow as show below.
```
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
```

### Results
The predictive model reached a Mean Absolute Error (MAE) of $1.50 when predicting average fare. Below shows a series of test used to validate where circles are the actual average fair and x's are the predicted value.
![preditions_vs_actual_data](https://github.com/user-attachments/assets/139f0980-1588-4173-93e1-b5c8f05f5f8e)

Below is a graph of data represening the use of this function for all 168 combinations of weekdays and hours
![predicted_profit_line_graph](https://github.com/user-attachments/assets/e0f9c1d5-b12b-4786-9fd0-34b745ce57fb)
From this graph we can see that on average the highest average fairs are seen from 3am-6am for all days of the week

### Recomendations
Based on the analysis I recommened the following actions:
1. More drivers should be employed during the times of 1PM - 8PM on weekdays specificly because these are peak hours.
2. Those looking to maximize their profits per fair should work between 3AM - 6PM (mostlikely due to surcharges).
3. Try raising rates on Sunday and Saturday to account for less ride volume which could bring weekend profits inline with weekdays.

### Limitations
1. The model does not account for (weather conditions, surcharges, fair distance)
2. The dataset had to be significantly reduced due to the size therfore this entire project is based of data from December 2023 specifically which could have scewed the data.

### References
Dataset: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
