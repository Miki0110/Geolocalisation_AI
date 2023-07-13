import os
import numpy as np
from geopy.geocoders import Nominatim

# Function to get the coordinates of a country
def get_country_coordinates(country):
    geolocator = Nominatim(user_agent="country_converter")
    location = geolocator.geocode(country, exactly_one=True)

    if location is None:
        return None

    latitude = location.latitude
    longitude = location.longitude
    return latitude, longitude


if __name__ == "__main__":
    # Folder path(s)
    folder_path = [r"C:\Users\mikip\Pictures\50k_countryonly"]

    folder_names = []
    for path in folder_path:
        # Get the folder names
        folder_names += os.listdir(path)

    country_cord = {}
    # Get the coordinates of each folder
    for folder_name in folder_names:
        coords = get_country_coordinates(folder_name)
        print(folder_name, coords)

        if coords is None:
            continue

        country_cord[folder_name] = coords

    # Save the coordinates and names to a file
    np.save("country_cord.npy", country_cord)