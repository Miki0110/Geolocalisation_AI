import os
import numpy as np
from geopy.geocoders import Nominatim
from utils.dataLoaders.geoLocationDataset import GeoLocationDataset

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
    folder_path = r"C:\Users\Muku\OneDrive - Aalborg Universitet\Geo_sets\50k_country_only"
    country_cord = {}
    dataset = GeoLocationDataset(root_dir=folder_path, transform=None)
    # Get the coordinates of each folder
    for folder_name in dataset.dataset.classes:
        coords = get_country_coordinates(folder_name)
        print(folder_name, coords)

        if coords is None:
            continue

        country_cord[folder_name] = coords

    # Save the coordinates and names to a file
    np.save("country_cord.npy", country_cord)