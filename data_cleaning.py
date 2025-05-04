import pandas as pd
import sys

# Load your original CSV file
def add_id_column(df):
    # Add an 'id' column starting from 1
    df['id'] = range(1, len(df) + 1)
    # Optionally move 'id' to the front
    df = df[['id'] + [col for col in df.columns if col != 'id']]
    df.to_csv("cleaned_taxi_data_with_id.csv", index=False)


# Drop all rows where distance is 0 or total_amount is 0
def drop_zeros(df):
    print(f"Number of rows before dropping zeros: {len(df)}")
    df = df[(df['trip_distance'] != 0) & (df['total_amount'] != 0)]
    print(f"Number of rows after dropping zeros: {len(df)}")
    df.to_csv("cleaned_taxi_data_no_zeros.csv", index=False)

#  Add column for day of the week
def add_day_of_week(df):
    print("Adding day of the week column")
    # Covert data to pandas datetime format
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce') 
    df['day_of_week'] = df['tpep_pickup_datetime'].dt.day_name() #use the date time column to get day of the week
    df.to_csv("cleaned_taxi_data_with_day_of_week.csv", index=False)

# Calculate the tip percentage
def add_tip_percent(df):
    print("Adding tip percentage column")
    df['tip_percent'] = (df['tip_amount'] / df['total_amount']) * 100
    df.to_csv("cleaned_taxi_data_with_tip_percent.csv", index=False)


if __name__ == "__main__":
    func_name = sys.argv[1]
    if func_name == "add_id_column":
        df = pd.read_csv("cleaned_taxi_data.csv")
        add_id_column(df)
    if func_name == "drop_zeros":
        df = pd.read_csv("cleaned_taxi_data_with_id.csv")
        drop_zeros(df)
    if func_name == "add_day_of_week":
        df = pd.read_csv("cleaned_taxi_data_no_zeros.csv")
        add_day_of_week(df)
    if func_name == "add_tip_percent":
        df = pd.read_csv("cleaned_taxi_data_with_tip.csv")
        add_tip_percent(df)
    else:
        print(f"Function {func_name} not recognized.")
