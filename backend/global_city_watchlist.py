
"""
Global City Watchlist
---------------------
Manages monitored cities and automatically resolves coordinates
for existing starter cities that do not yet have coordinates.
"""

import json
from pathlib import Path

from city_weather import search_city


PROJECT_ROOT = Path(__file__).resolve().parent.parent

WATCHLIST_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "global_city_watchlist.json"
)


STARTER_CITIES = [
    {"city": "Hyderabad", "country": "India"},
    {"city": "Mumbai", "country": "India"},
    {"city": "Delhi", "country": "India"},
    {"city": "Tokyo", "country": "Japan"},
    {"city": "Beijing", "country": "China"},
    {"city": "London", "country": "United Kingdom"},
    {"city": "New York", "country": "United States"},
    {"city": "Sao Paulo", "country": "Brazil"},
    {"city": "Cairo", "country": "Egypt"},
    {"city": "Sydney", "country": "Australia"},
]


def _city_key(city):
    return (
        city.get("city", "").strip().casefold(),
        city.get("country", "").strip().casefold()
    )


def _save_watchlist(cities):
    WATCHLIST_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        WATCHLIST_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            cities,
            file,
            indent=4,
            ensure_ascii=False
        )


def _resolve_city(city):
    """
    Find coordinates and timezone for a city using
    Open-Meteo geocoding.

    Existing coordinates are preserved.
    """

    if (
        city.get("latitude") is not None
        and city.get("longitude") is not None
    ):
        return city

    city_name = city.get("city", "").strip()
    expected_country = city.get("country", "").strip().casefold()

    matches = search_city(
        city_name,
        result_count=10
    )

    if not matches:
        raise ValueError(
            f"No geocoding result found for {city_name}."
        )

    # Prefer a result whose country matches the saved country.
    selected = None

    for result in matches:

        result_country = (
            result.get("country", "").strip().casefold()
        )

        if result_country == expected_country:
            selected = result
            break

    # If no country match is found, use the first result.
    if selected is None:
        selected = matches[0]

        print(
            f"Warning: country match not found for "
            f"{city_name}. Using first geocoding result."
        )

    resolved = {
        "city": selected.get("name", city_name),
        "country": selected.get("country", city.get("country", "")),
        "admin1": selected.get("admin1", ""),
        "latitude": float(selected["latitude"]),
        "longitude": float(selected["longitude"]),
        "timezone": selected.get("timezone", "auto")
    }

    return resolved


def load_watchlist():
    """
    Load saved cities.

    Creates the starter list if needed.
    Resolves missing coordinates and saves the updated list.
    """

    if not WATCHLIST_FILE.exists():

        cities = STARTER_CITIES.copy()
        _save_watchlist(cities)

    else:

        try:
            with open(
                WATCHLIST_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                cities = json.load(file)

        except (json.JSONDecodeError, OSError) as error:

            raise RuntimeError(
                f"Could not load the city watchlist: {error}"
            ) from error

        if not isinstance(cities, list):
            raise ValueError(
                "The watchlist JSON must contain a list."
            )

    updated_cities = []
    changed = False

    print("\nChecking city coordinates...")

    for city in cities:

        try:

            resolved = _resolve_city(city)

            updated_cities.append(resolved)

            if resolved != city:
                changed = True

            print(
                f"  {resolved['city']}, "
                f"{resolved['country']}: "
                f"{resolved['latitude']}, "
                f"{resolved['longitude']}"
            )

        except Exception as error:

            print(
                f"  Could not resolve "
                f"{city.get('city', 'Unknown')}: {error}"
            )

            # Keep the city in the watchlist so it can
            # be retried during a later load.
            updated_cities.append(city)

    if changed:
        _save_watchlist(updated_cities)
        print("\nUpdated watchlist coordinates saved.")

    return updated_cities


def add_city(city_name):
    """Search for a city and add it to the watchlist."""

    city_name = city_name.strip()

    if not city_name:
        raise ValueError("Please enter a city name.")

    matches = search_city(
        city_name,
        result_count=10
    )

    if not matches:
        raise ValueError(
            f"No matching city found for '{city_name}'."
        )

    # Use the first result for this command-line version.
    result = matches[0]

    city = {
        "city": result.get("name", city_name),
        "country": result.get("country", ""),
        "admin1": result.get("admin1", ""),
        "latitude": float(result["latitude"]),
        "longitude": float(result["longitude"]),
        "timezone": result.get("timezone", "auto")
    }

    cities = load_watchlist()

    if any(
        _city_key(existing) == _city_key(city)
        for existing in cities
    ):
        raise ValueError(
            f"{city['city']}, {city['country']} "
            "is already in the watchlist."
        )

    cities.append(city)
    _save_watchlist(cities)

    return city


def remove_city(city_name, country=None):
    """Remove a city by name and optional country."""

    cities = load_watchlist()

    target_name = city_name.strip().casefold()

    target_country = (
        country.strip().casefold()
        if country
        else None
    )

    updated_cities = [
        city
        for city in cities
        if not (
            city.get("city", "").strip().casefold()
            == target_name
            and (
                target_country is None
                or city.get("country", "").strip().casefold()
                == target_country
            )
        )
    ]

    if len(updated_cities) == len(cities):
        raise ValueError(
            f"City '{city_name}' was not found."
        )

    _save_watchlist(updated_cities)

    return updated_cities


def display_watchlist():
    """Display all saved cities and their coordinates."""

    cities = load_watchlist()

    print("\nGLOBAL WEATHER MONITORING WATCHLIST")
    print("-" * 60)

    for index, city in enumerate(cities, start=1):

        region = city.get("admin1", "")

        location = ", ".join(
            part
            for part in [
                city.get("city", ""),
                region,
                city.get("country", "")
            ]
            if part
        )

        latitude = city.get("latitude", "Missing")
        longitude = city.get("longitude", "Missing")

        print(
            f"{index}. {location} | "
            f"Coordinates: {latitude}, {longitude}"
        )

    print("-" * 60)
    print(f"Total monitored cities: {len(cities)}")


if __name__ == "__main__":

    print("GLOBAL CITY WATCHLIST TEST")

    display_watchlist()

    print("\nGLOBAL CITY WATCHLIST TEST COMPLETED")
