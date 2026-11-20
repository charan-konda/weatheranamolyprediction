
import requests
import pandas as pd
from pathlib import Path

# Open-Meteo API endpoint
API_URL = "https://api.open-meteo.com/v1/forecast"

# Five project locations
LOCATIONS = [
    {"location_id": 0, "location_name": "Visakhapatnam",
     "latitude": 17.6868, "longitude": 83.2185},

    {"location_id": 1, "location_name": "Kakinada",
     "latitude": 16.9891, "longitude": 82.2475},

    {"location_id": 2, "location_name": "Machilipatnam",
     "latitude": 16.1809, "longitude": 81.1303},

    {"location_id": 3, "location_name": "Nellore",
     "latitude": 14.4426, "longitude": 79.9865},

    {"location_id": 4, "location_name": "Ongole",
     "latitude": 15.5057, "longitude": 80.0499}
]

# Same output variables as our historical dataset
CURRENT_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "pressure_msl",
    "wind_gusts_10m"
]

# Project root and output file
PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_FILE = (
    PROJECT_ROOT / "datasets" / "raw" / "live_weather.csv"
)


def fetch_live_weather():

    print("Connecting to Open-Meteo API...")

    # Prepare coordinates for all five locations
    latitudes = ",".join(
        str(location["latitude"])
        for location in LOCATIONS
    )

    longitudes = ",".join(
        str(location["longitude"])
        for location in LOCATIONS
    )

    params = {
        "latitude": latitudes,
        "longitude": longitudes,
        "current": ",".join(CURRENT_VARIABLES),
        "timezone": "Asia/Kolkata",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm"
    }

    try:
        response = requests.get(
            API_URL,
            params=params,
            timeout=30
        )

        response.raise_for_status()
        data = response.json()

        # Multiple coordinates return multiple location objects
        if isinstance(data, dict):
            data = [data]

        records = []

        for location, weather in zip(LOCATIONS, data):

            current = weather["current"]

            record = {
                "location_id": location["location_id"],
                "location_name": location["location_name"],
                "time": current["time"],
                "temperature_2m_c": current["temperature_2m"],
                "relative_humidity_2m_pct":
                    current["relative_humidity_2m"],
                "precipitation_mm": current["precipitation"],
                "wind_speed_10m_kmh": current["wind_speed_10m"],
                "wind_direction_10m_deg":
                    current["wind_direction_10m"],
                "pressure_msl_hpa": current["pressure_msl"],
                "wind_gusts_10m_kmh": current["wind_gusts_10m"]
            }

            records.append(record)

        if len(records) != len(LOCATIONS):
            raise ValueError(
                "API did not return data for all five locations."
            )

        live_df = pd.DataFrame(records)

        # Create output directory if it does not exist
        OUTPUT_FILE.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # Append new readings without duplicating location/time pairs
        if OUTPUT_FILE.exists():

            old_df = pd.read_csv(OUTPUT_FILE)

            live_df = pd.concat(
                [old_df, live_df],
                ignore_index=True
            )

        live_df = live_df.drop_duplicates(
            subset=["location_id", "time"],
            keep="last"
        )

        live_df = live_df.sort_values(
            ["location_id", "time"]
        )

        live_df.to_csv(
            OUTPUT_FILE,
            index=False
        )

        print("\nLive weather retrieved successfully!")

        print("Locations received:", len(records))
        print("Saved file:", OUTPUT_FILE)

        print("\nLatest readings:")
        print(pd.DataFrame(records).to_string(index=False))

        return pd.DataFrame(records)

    except requests.exceptions.RequestException as error:
        print("\nAPI connection error:", error)

    except (KeyError, TypeError, ValueError) as error:
        print("\nData processing error:", error)

    return None


if __name__ == "__main__":
    fetch_live_weather()
