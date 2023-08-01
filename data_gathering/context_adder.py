import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import pandas as pd
import os
import random
import shutil
from geopy.geocoders import Nominatim
import numpy as np


def get_country_name(location):
    geolocator = Nominatim(user_agent="geoapiExercises")

    # getting the location
    location = geolocator.geocode(location, timeout=10, language='en')

    if location is None:
        return None

    # return country name
    return location.address.split(",")[-1].strip()

class ContextAdder:
    def __init__(self, root, dir_path):
        self.root = root
        self.dir_path = dir_path
        self.images = self.get_all_images()

        self.data = pd.read_csv('image_data.csv', sep=',')
        self.index = 0
        self.road_vars = {col: tk.IntVar() for col in self.data.columns[4:14]}  # start from 4th column
        self.bg_vars = {col: tk.IntVar() for col in self.data.columns[14:]}  # start from 14th column

        # Create a frame for the image
        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        # Create a frame for the checkboxes
        self.road_frame = tk.Frame(self.root)
        self.road_frame.pack()

        for col, var in self.road_vars.items():
            cb = tk.Checkbutton(self.road_frame, text=col, variable=var)
            cb.pack(side="left")

        self.bg_frame = tk.Frame(self.root)
        self.bg_frame.pack()

        for col, var in self.bg_vars.items():
            cb = tk.Checkbutton(self.bg_frame, text=col, variable=var)
            cb.pack(side="left")

        # Create a frame for the buttons
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack()

        self.previous_button = tk.Button(self.button_frame, text="Previous Image", command=self.previous_image)
        self.previous_button.grid(row=0, column=0)

        self.next_button = tk.Button(self.button_frame, text="Next Image", command=self.next_image)
        self.next_button.grid(row=0, column=1)

        self.save_button = tk.Button(self.root, text="Save Changes", command=self.save_changes)
        self.save_button.pack(side="right")

        # Create a label to display the position
        self.position_label = tk.Label(root, text="")
        self.position_label.pack(side="bottom")

        # Create slider (scale)
        self.slider = tk.Scale(root, from_=0, to=len(self.images)-1, orient='horizontal', command=self.on_slider_change)
        self.slider.pack()

        self.load_image()

    def get_all_images(self):
        images = []
        for subdir, dirs, files in os.walk(self.dir_path):
            for file in files:
                # Check if the file is an image
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    images.append(os.path.join(subdir, file))
        random.shuffle(images)
        return images

    def load_image(self):
        # Use the original directory to load the image
        image_path = self.images[self.index]
        img = Image.open(image_path)
        img = img.resize((880, 400), Image.LANCZOS)  # resize the image
        img = ImageTk.PhotoImage(img)

        self.image_label.config(image=img)
        self.image_label.image = img

        # Check if image_path already exists in self.data
        new_path = os.path.join('images', os.path.basename(image_path))
        if new_path in self.data['image_path'].values:
            image_data = self.data.loc[self.data['image_path'] == new_path]
            # If it exists, load previously stored data into checkboxes
            for col, var in self.road_vars.items():
                var.set(image_data[col].values[0])

            for col, var in self.bg_vars.items():
                var.set(image_data[col].values[0])
        else:
            # If not, clear all checkboxes
            for var in self.road_vars.values():
                var.set(0)
            for var in self.bg_vars.values():
                var.set(0)
        self.slider.set(self.index)  # Set slider value to current image index
        self.update_position_label()

    def next_image(self):
        self.save_current_scores()
        self.index += 1
        if self.index >= len(self.data):
            self.index = 0  # loop back to the beginning
        self.load_image()

    def previous_image(self):
        self.index -= 1
        if self.index < 0:  # To prevent going to negative indices
            self.index = 0
        self.load_image()  # Load the image of the new index

    def on_slider_change(self, value):
        # Slider (scale) callback to jump between images
        self.index = int(value)
        self.load_image()
        self.update_position_label()

    def update_position_label(self):
        # Update the position label to show the current position
        self.position_label.config(text=f"Image {self.index + 1} of {len(self.images)}")

    def save_current_scores(self):
        # Use the original directory when working with the image
        image_path = self.images[self.index]

        # Extract the name of the folder from the image path
        folder_name = os.path.basename(os.path.dirname(image_path))
        # Convert it to the English name of the country
        country_name = get_country_name(folder_name)

        # Use the new directory when saving the image path in the CSV
        new_path = os.path.join('images', os.path.basename(image_path))

        # Initialize an empty scores dictionary with all columns set to some default value (like np.nan or None)
        scores = dict.fromkeys(self.data.columns, np.nan)

        scores["image_path"] = new_path
        scores["country"] = country_name
        for col, var in self.road_vars.items():
            scores[col] = var.get()

        for col, var in self.bg_vars.items():
            scores[col] = var.get()

        # If image_path already exists in self.data, update that row; otherwise append a new row
        if new_path in self.data['image_path'].values:
            for col, value in scores.items():
                self.data.loc[self.data['image_path'] == new_path, col] = value
        else:
            self.data = pd.concat([self.data, pd.DataFrame([scores])], ignore_index=True)

        # After scoring the image, copy it to the 'images' directory
        shutil.copy(image_path, new_path)

        self.save_changes()

    def save_changes(self):
        self.data.to_csv('image_data.csv', index=False)


if __name__ == "__main__":
    dir_path = r'C:\Users\Muku\OneDrive - Aalborg Universitet\Geo_sets\50k_country_only'
    root = tk.Tk()
    app = ContextAdder(root, dir_path)
    root.mainloop()