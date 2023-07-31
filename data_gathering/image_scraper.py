import os
import random
import numpy as np
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains as AC
import atexit
from urllib.parse import urlparse
from geopy.geocoders import Nominatim
import tkinter as tk
from tkinter import messagebox


# List of IDs that clutter the webpage
clutter = ['titlecard', 'play', 'minimap']

# Road classes
road_classes = {1: "Highways (multiple lanes, long distance travel)",
                2: "Urban roads (roads in big cities, towns, villages)",
                3: "Country Roads (fewer lanes, near rural areas)",
                4: "Mountain passes (roads through mountainous areas, guard rails)",
                5: "Coastal roads (roads near the ocean, beaches, cliffs)",
                6: "Dirt roads (unpaved / gravel roads)",
                7: "Desert roads (roads through deserts, sand dunes)",
                8: "Tunnel (underground roads)",
                9: "Bridges (roads that cross over water)",
                10: "None (no roads in the image)"
                }
# Background classes
background_classes = {1: "Urban (Cityscapes, high-rise buildings, busy streets)",
                      2: "Suburban (Houses, small buildings, quiet streets)",
                      3: "Rural (Fields, farms, forests, mountains, lakes)",
                      4: "Mountainous (Mountains, hills, cliffs, valleys)",
                      5: "Coastal (Beaches, oceans, lakes, rivers, boats)",
                      6: "Forest (Primary vegetation is trees, forests, jungles)",
                      7: "Desert (Sandy areas, sand dunes, cacti, dry areas)",
                      8: "Wetlands (Swamps, marshes, bogs, wet areas)",
                      9: "Snowy (Snow, ice, glaciers, frozen lakes)",
                      10: "Agricultural (Farms, fields, crops, livestock)"
                      }


def get_classes_via_gui(classes_dict):
    # Create a new window
    root = tk.Tk()

    # Create a list to store the checkbutton variables
    check_vars = []

    # variable to store whether the image is skipped
    skip_image = tk.BooleanVar(value=False)

    # List to store the selected classes
    selected_classes = []

    # function to get and store selected checkboxes
    def get_selected_classes():
        nonlocal selected_classes
        selected_classes = []
        for i, check_var in enumerate(check_vars):
            if check_var.get() == 1:
                selected_classes.append(1)
            else:
                selected_classes.append(0)
        root.destroy()

    def skip():
        skip_image.set(True)
        root.destroy()

    # Add a checkbutton for each class
    for idx, road_class in classes_dict.items():
        check_var = tk.IntVar()
        check_vars.append(check_var)
        check = tk.Checkbutton(root, text=road_class, variable=check_var)
        check.pack()

    # Add a button that will store the selected classes and destroy the window when clicked
    button = tk.Button(root, text="OK", command=get_selected_classes)
    button.pack()

    # Add a 'Skip' button
    skip_button = tk.Button(root, text="Skip", command=skip)
    skip_button.pack()

    root.mainloop()

    if skip_image.get():
        return None
    else:
        return selected_classes

# Get a random world coord, numbers based on me trying to avoid the ocean
def get_random_coordinates():
    # Read the data (use the first 19 columns, as some city names include tabs)
    df = pd.read_csv('cities15000.txt', sep='\t', header=None, usecols=range(19), low_memory=False)

    # Define the column names
    column_names = ['geonameid', 'name', 'asciiname', 'alternatenames', 'latitude', 'longitude',
                    'feature class', 'feature code', 'country code', 'cc2', 'admin1 code',
                    'admin2 code', 'admin3 code', 'admin4 code', 'population', 'elevation',
                    'dem', 'timezone', 'modification date']

    df.columns = column_names

    # Remove rows where 'country code' is NaN
    df = df.dropna(subset=['country code'])

    random_city = df.sample()

    return random_city['latitude'].values[0], random_city['longitude'].values[0]


# Registers for exit
def save_data_on_exit():
    if data_list:
        # Check if file exists
        csv_exists = os.path.isfile('image_data.csv')

        df = pd.DataFrame(data_list)

        # If file exists, append without headers
        if csv_exists:
            df.to_csv('image_data.csv', mode='a', header=False, index=False)
        else:  # Otherwise, write with headers
            df.to_csv('image_data.csv', mode='w', header=True, index=False)

        print('Data saved!')


def quit_driver():
    if driver:
        driver.quit()
        print("Driver closed!")


# Register the functions to be called on exit
atexit.register(save_data_on_exit)
atexit.register(quit_driver)


def get_next_filename(image_dir, country):
    # Get all files in the image_dir
    files = list(Path(image_dir).glob(f"{country}_*.png"))

    if not files:  # if no files found for this country
        return f"{country}_0.png"

    # extract numbers from filenames
    numbers = [int(file.stem.split('_')[-1]) for file in files]

    # Get the highest number
    highest_number = max(numbers)

    # Return the filename with the highest_number + 1
    return f"{country}_{highest_number + 1}.png"


if __name__ == "__main__":
    geolocator = Nominatim(user_agent="my-app")
    # Get current folder
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    # initialize webdriver
    webdriver_path = os.path.join(curr_dir, 'chromedriver.exe')
    driver = webdriver.Chrome()

    # URL
    URL = 'https://www.google.com/maps/'

    # List to store the data
    data_list = []

    # Define the number of images you want to download
    num_images = 1000

    # Directory where you want to store the images
    image_dir = "images"
    os.makedirs(image_dir, exist_ok=True)

    for i in range(num_images):
        lat, lon = get_random_coordinates()
        driver.get(URL + f'@{lat},{lon},6z')
        # Wait for the page to load
        try:
            element_present = EC.presence_of_element_located((By.CLASS_NAME, 'q2sIQ'))
            WebDriverWait(driver, 5).until(element_present)
        except Exception as e:
            print(f"Timed out waiting for page to load: {str(e)}")

        # Find pegman (Street View icon)
        pegman = driver.find_element(By.CLASS_NAME, 'q2sIQ')

        # Adjust these numbers according to your screen resolution
        window_size = driver.get_window_size()
        screen_width = window_size['width']
        screen_height = window_size['height']

        # Generate random location on screen for drop
        center_x = screen_width // 2 - 200
        center_y = screen_height // 2 - 25

        drop_x_range = (center_x - 100, center_x + 100)
        drop_y_range = (center_y - 50, center_y + 50)

        drop_x = -random.randint(*drop_x_range)
        drop_y = -random.randint(*drop_y_range)

        # Perform drag and drop action
        AC(driver).drag_and_drop_by_offset(pegman, drop_x, drop_y).perform()
        # Wait for the URL to change
        try:
            WebDriverWait(driver, 3).until(EC.url_contains("/data=!"))
        except Exception as e:
            print(f"Timed out waiting for URL to change: {str(e)}")
            continue
        # Hide the adds and other clutter
        for element_id in clutter:
            try:
                element = driver.find_element(By.ID, element_id)
                driver.execute_script("arguments[0].style.visibility='hidden';", element)
            except Exception as e:
                print(f"Could not hide element with id '{element_id}'. Error: {str(e)}")
        # parse the page content
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Get the coordinates from the URL
        path = urlparse(driver.current_url).path
        lat, lon = path.split('/')[2].split('@')[1].split(',')[0:2]
        print(lat, lon)

        # extract location info
        location = geolocator.reverse(f"{lat}, {lon}", exactly_one=True, language='en', timeout=10)
        if location is None:
            continue

        # get user input for classes via GUI
        print("Classification of road types")
        road_input = get_classes_via_gui(road_classes)
        if road_input is None:
            print("Skipping image...")
            continue

        print("Classification of background types")
        background_input = get_classes_via_gui(background_classes)
        if background_input is None:
            print("Skipping image...")
            continue

        # Get the country from the location
        country = location.raw['address'].get('country', '')
        print(country)
        # define a filename based on the country and coordinates
        filename = get_next_filename(image_dir, country)
        filename = os.path.join(image_dir, filename)

        # take a screenshot and save the image
        driver.save_screenshot(filename)

        # fill in the data list
        data = {
            'country': country,
            'image_path': filename,
            'latitude': lat,
            'longitude': lon
        }

        for idx, val in enumerate(road_input, start=1):
            data[road_classes[idx]] = val

        for idx, val in enumerate(background_input, start=1):
            data[background_classes[idx]] = val

        data_list.append(data)
