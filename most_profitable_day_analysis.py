import matplotlib.pyplot as plt
import pandas as pd

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

bars = plt.bar(days, averages, color=['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink'])
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,   # X position (center of bar)
        height + 0.3,                        # Y position (just above the bar)
        f"{height:.2f}",                     # Text label (2 decimal places)
        ha='center', va='bottom', fontsize=9
    )
plt.ylabel("Average Total Amount ($)")
plt.title("Average Total Fare by Day of Week")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()