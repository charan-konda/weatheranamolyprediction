"""
City search and weather retrieval using Open-Meteo.

This module preserves the existing current-weather functions and adds
24-hour hourly weather retrieval for the spatio-temporal GNN pipeline.

The hourly function returns the same eight feature names used during
ST-GNN training. It does not modify the historical training dataset.
"""

import json
from datetime import datetime
from urllib.parse import urlencode
from urllib.request import Request, urlopen


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# Existing reference locations used to train the first ST-GNN checkpoint.
REFERENCE_LOCATIONS = [
    {"name": "Visakhapatnam", "latitude": 17.6868, "longitude": 83.2185},
    {"name": "Kakinada", "latitude": 16.9891, "longitude": 82.2475},
    {"name": "Machilipatnam", "latitude": 16.1809, "longitude": 81.1303},
    {"name": "Nellore", "latitude": 14.4426, "longitude": 79.9865},
    {"name": "Ongole", "latitude": 15.5057, "longitude": 80.0499},
]

CURRENT_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "pressure_msl",
    "wind_gusts_10m",
]

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "pressure_msl",
    "wind_gusts_10m",
]

# Must match the feature order used by models/train_stgnn.py.
STGNN_FEATURE_COLUMNS = [
    "temperature_2m_c",
    "relative_humidity_2m_pct",
    "precipitation_mm",
    "wind_speed_10m_kmh",
    "pressure_msl_hpa",
    "wind_gusts_10m_kmh",
    "wind_sin",
    "wind_cos",
]


def _get_json(url, params):
    """Send a GET request and return the decoded JSON response."""
    query_string = urlencode(params)
    full_url = f"{url}?{query_string}"

    request = Request(
        full_url,
        headers={"User-Agent": "Weather-Anomaly-Research-Prototype/1.0"},
    )

    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as error:
        raise RuntimeError(
            f"Weather API request failed: {error}"
        ) from error


def search_city(city_name, result_count=5):
    """
    Search for cities by name.

    Returns dictionaries containing name, country, admin1, coordinates,
    and timezone. This preserves the interface used by the watchlist code.
    """
    city_name = str(city_name).strip()

    if len(city_name) < 2:
        raise ValueError("Enter at least two characters for the city name.")

    data = _get_json(
        GEOCODING_URL,
        {
            "name": city_name,
            "count": result_count,
            "language": "en",
            "format": "json",
        },
    )

    cities = []
    for item in data.get("results", []):
        if "latitude" not in item or "longitude" not in item:
            continue

        cities.append(
            {
                "name": item.get("name", "Unknown"),
                "country": item.get("country", "Unknown"),
                "admin1": item.get("admin1", ""),
                "latitude": float(item["latitude"]),
                "longitude": float(item["longitude"]),
                "timezone": item.get("timezone", "auto"),
            }
        )

    return cities


def _validate_coordinates(latitude, longitude):
    latitude = float(latitude)
    longitude = float(longitude)

    if not -90 <= latitude <= 90:
        raise ValueError("Latitude must be between -90 and 90.")
    if not -180 <= longitude <= 180:
        raise ValueError("Longitude must be between -180 and 180.")

    return latitude, longitude


def _require_values(record, required, context):
    missing = [
        field for field in required
        if record.get(field) is None
    ]
    if missing:
        raise RuntimeError(
            f"{context} is missing required values: " + ", ".join(missing)
        )


def _to_feature_row(timestamp, values):
    """Convert an Open-Meteo hourly record to the trained model features."""
    _require_values(
        values,
        HOURLY_VARIABLES,
        f"Hourly weather record at {timestamp}",
    )

    direction = float(values["wind_direction_10m"])
    direction_radians = direction * 3.141592653589793 / 180.0

    return {
        "time": timestamp,
        "temperature_2m_c": float(values["temperature_2m"]),
        "relative_humidity_2m_pct": float(values["relative_humidity_2m"]),
        "precipitation_mm": float(values["precipitation"]),
        "wind_speed_10m_kmh": float(values["wind_speed_10m"]),
        "pressure_msl_hpa": float(values["pressure_msl"]),
        "wind_gusts_10m_kmh": float(values["wind_gusts_10m"]),
        "wind_sin": __import__("math").sin(direction_radians),
        "wind_cos": __import__("math").cos(direction_radians),
        "wind_direction_10m_deg": direction,
    }


def fetch_current_weather(latitude, longitude, timezone="auto"):
    """
    Retrieve current weather for one geographic location.

    Units: Celsius, percent, mm, km/h, degrees, and hPa.
    """
    latitude, longitude = _validate_coordinates(latitude, longitude)

    data = _get_json(
        WEATHER_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "current": ",".join(CURRENT_VARIABLES),
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
            "precipitation_unit": "mm",
            "timezone": timezone or "auto",
        },
    )

    current = data.get("current")
    if not current:
        raise RuntimeError(
            "The weather API did not return current conditions."
        )

    required = ["time"] + CURRENT_VARIABLES
    _require_values(current, required, "Current weather response")

    direction = float(current["wind_direction_10m"])
    import math

    radians = math.radians(direction)

    return {
        "time": current["time"],
        "temperature_2m_c": float(current["temperature_2m"]),
        "relative_humidity_2m_pct": float(current["relative_humidity_2m"]),
        "precipitation_mm": float(current["precipitation"]),
        "wind_speed_10m_kmh": float(current["wind_speed_10m"]),
        "wind_direction_10m_deg": direction,
        "pressure_msl_hpa": float(current["pressure_msl"]),
        "wind_gusts_10m_kmh": float(current["wind_gusts_10m"]),
        "wind_sin": math.sin(radians),
        "wind_cos": math.cos(radians),
    }


def fetch_recent_hourly_weather(
    latitude,
    longitude,
    timezone="auto",
    hours=24,
):
    """
    Fetch the most recent complete hourly weather sequence for a city.

    The API request retrieves a small historical/future buffer. This
    function discards future timestamps and returns the last `hours`
    available hourly records up to the latest current local hour.

    Each returned row contains `time` plus the eight ST-GNN feature
    columns, as well as wind_direction_10m_deg for inspection.

    Raises an error rather than silently returning an incomplete sequence.
    """
    latitude, longitude = _validate_coordinates(latitude, longitude)

    hours = int(hours)
    if hours < 1 or hours > 168:
        raise ValueError("hours must be between 1 and 168.")

    data = _get_json(
        WEATHER_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m",
            "hourly": ",".join(HOURLY_VARIABLES),
            "past_days": 2,
            "forecast_days": 1,
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
            "precipitation_unit": "mm",
            "timezone": timezone or "auto",
        },
    )

    hourly = data.get("hourly")
    if not hourly or not hourly.get("time"):
        raise RuntimeError("The weather API returned no hourly observations.")

    times = hourly["time"]
    available_lengths = [
        len(hourly.get(variable, []))
        for variable in HOURLY_VARIABLES
    ]
    if any(length != len(times) for length in available_lengths):
        raise RuntimeError(
            "The hourly API response contains inconsistent array lengths."
        )

    # Open-Meteo returns local timestamps in the requested timezone.
    # Compare timestamps lexically because they use sortable ISO local format.
    current_record = data.get("current") or {}
    api_current_time = current_record.get("time")

    if not api_current_time:
        raise RuntimeError(
            "The weather API did not return its current local timestamp."
        )

    # Use the API's location-local clock, not the computer's timezone.
    now_local_hour = str(api_current_time)[:13] + ":00"

    eligible_indices = [
        index for index, timestamp in enumerate(times)
        if timestamp <= now_local_hour
    ]

    selected_indices = eligible_indices[-hours:]
    if len(selected_indices) != hours:
        raise RuntimeError(
            f"Only {len(selected_indices)} hourly records were available; "
            f"{hours} are required for ST-GNN inference."
        )

    rows = []
    for index in selected_indices:
        values = {
            variable: hourly[variable][index]
            for variable in HOURLY_VARIABLES
        }
        rows.append(_to_feature_row(times[index], values))

    # Verify chronological order and exact sequence length.
    if len(rows) != hours:
        raise RuntimeError("The hourly sequence has an unexpected length.")

    return rows


def get_city_weather(city):
    """Retrieve current weather for a geocoded watchlist city."""
    weather = fetch_current_weather(
        city["latitude"],
        city["longitude"],
        city.get("timezone", "auto"),
    )

    return {
        "city": city.get("city", city.get("name", "")),
        "country": city.get("country", ""),
        "admin1": city.get("admin1", ""),
        "latitude": float(city["latitude"]),
        "longitude": float(city["longitude"]),
        **weather,
    }


def get_city_hourly_weather(city, hours=24):
    """Retrieve a city's recent hourly sequence with watchlist metadata."""
    rows = fetch_recent_hourly_weather(
        city["latitude"],
        city["longitude"],
        city.get("timezone", "auto"),
        hours=hours,
    )

    city_info = {
        "city": city.get("city", city.get("name", "")),
        "country": city.get("country", ""),
        "admin1": city.get("admin1", ""),
        "latitude": float(city["latitude"]),
        "longitude": float(city["longitude"]),
        "timezone": city.get("timezone", "auto"),
    }

    return [
        {**city_info, **row}
        for row in rows
    ]


if __name__ == "__main__":
    print("=" * 64)
    print("CITY SEARCH AND HOURLY WEATHER API TEST")
    print("=" * 64)

    test_city_name = "Hyderabad"
    print(f"\nSearching for: {test_city_name}")

    try:
        matches = search_city(test_city_name)

        if not matches:
            raise RuntimeError("No matching city was found.")

        selected_city = matches[0]
        print(
            "Selected:",
            selected_city["name"],
            selected_city["admin1"],
            selected_city["country"],
        )

        print("\nFetching current weather...")
        current = get_city_weather(selected_city)
        print("Current weather timestamp:", current["time"])
        print("Temperature:", current["temperature_2m_c"], "°C")

        print("\nFetching the latest 24 hourly records...")
        hourly_rows = get_city_hourly_weather(selected_city, hours=24)
        print("Hourly records:", len(hourly_rows))
        print("First timestamp:", hourly_rows[0]["time"])
        print("Last timestamp:", hourly_rows[-1]["time"])
        print("ST-GNN feature columns:")
        for column in STGNN_FEATURE_COLUMNS:
            print(f"  {column}: {hourly_rows[-1][column]}")

        print("\nCITY WEATHER API TEST PASSED")

    except Exception as error:
        print("\nCITY WEATHER API TEST FAILED")
        print(error)
        raise SystemExit(1)
