import os
import time
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import atexit
import pycountry

clutter = ['address', 'adnotice', 'ad', 'controls', 'map_canvas', 'controls', 'share', 'aswift_1_host', "minimaximize"]

landscape_classes = {1: "Mountain/hill",
                     2: "Forest/wood",
                     3: "Desert",
                     4: "Beach/Coast",
                     5: "Plains",
                     6: "Urban",
                     7: "Sub-Urban",
                     8: "Rural",
                     9: "small road",
                     10: "Highway",
                     11: "Snow/Ice"}


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
atexit.register(save_data_on_exit)


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


def translate_country_name(name):
    try:
        country = pycountry.countries.search_fuzzy(name)
        # It returns a list of possible countries, we just take the first one as most likely
        return country[0].name
    except LookupError:
        print(f"Could not translate country name: {name}")
        return name


# Get current folder
curr_dir = os.path.dirname(os.path.abspath(__file__))
# initialize webdriver
driver = webdriver.Chrome(os.path.join(curr_dir, 'chromedriver.exe'))

# URL of the website
url = "https://randomstreetview.com/"

# List to store the data
data_list = []

# Define the number of images you want to download
num_images = 1000

# Directory where you want to store the images
image_dir = "images"
os.makedirs(image_dir, exist_ok=True)

for i in range(num_images):
    driver.get(url)
    # Wait for the page to load
    try:
        element_present = EC.presence_of_element_located((By.ID, clutter[0]))
        WebDriverWait(driver, 5).until(element_present)
    except Exception as e:
        print(f"Timed out waiting for page to load: {str(e)}")

    # Hide the adds and other clutter
    for element_id in clutter:
        try:
            element = driver.find_element_by_id(element_id)
            driver.execute_script("arguments[0].style.visibility='hidden';", element)
        except Exception as e:
            print(f"Could not hide element with id '{element_id}'. Error: {str(e)}")
    # parse the page content
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # get user input for classes
    classes = []
    for j in range(1, 12):  # 1 to 10
        class_input = input(f"Is it {landscape_classes[j]} (y,n): ")
        if class_input == "y":
            class_val = 1
        else:
            class_val = 0
        classes.append(class_val)

    # extract location info
    location_info = soup.find('div', id='address')
    if location_info is None:
        print("Could not find location info")
        continue
    country = location_info.text.split(',')[-1]
    country = translate_country_name(country.strip())

    # define a filename based on the country and coordinates
    filename = get_next_filename(image_dir, country)
    filename = os.path.join(image_dir, filename)

    # take a screenshot and save the image
    driver.save_screenshot(filename)

    # fill in the data list
    data_list.append({
        'country': country,
        'image_path': filename,
        landscape_classes[1]: classes[0],
        landscape_classes[2]: classes[1],
        landscape_classes[3]: classes[2],
        landscape_classes[4]: classes[3],
        landscape_classes[5]: classes[4],
        landscape_classes[6]: classes[5],
        landscape_classes[7]: classes[6],
        landscape_classes[8]: classes[7],
        landscape_classes[9]: classes[8],
        landscape_classes[10]: classes[9],
        landscape_classes[11]: classes[10],
    })

driver.quit()

# convert the list to a DataFrame and save as CSV
df = pd.DataFrame(data_list)
df.to_csv('image_data.csv', index=False)
