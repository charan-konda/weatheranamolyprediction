import pandas as pd

FILE_PATH = (
    "datasets/processed/"
    "andhra_pradesh_weather_cleaned.csv"
)

def load_historical_data():

    print("Loading historical weather data...")

    df = pd.read_csv(
        FILE_PATH,
        parse_dates=["time"]
    )

    df = df.sort_values(
        ["location_id", "time"]
    ).reset_index(drop=True)

    print("\nDataset loaded successfully!")

    print("Total records:", len(df))
    print("Total columns:", len(df.columns))

    print("\nLocations:")
    print(df["location_name"].unique())

    print("\nDate range:")
    print("Start:", df["time"].min())
    print("End:", df["time"].max())

    print("\nMissing values:")
    print(df.isnull().sum())

    return df


if __name__ == "__main__":

    weather_data = load_historical_data()

    print("\nFirst five records:")
    print(weather_data.head())