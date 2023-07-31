import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import pandas as pd


class ImageReviewer:

    def __init__(self, root):
        self.root = root
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
        self.position_label.pack(side="left")

        # Create slider (scale)
        self.slider = tk.Scale(root, from_=0, to=len(self.data) - 1, orient='horizontal', command=self.on_slider_change)
        self.slider.pack()

        self.load_image()

    def load_image(self):
        # Load the image of the current index
        image_path = self.data.iloc[self.index]['image_path']
        img = Image.open(image_path)
        img = img.resize((880, 400), Image.LANCZOS) # resize the image
        img = ImageTk.PhotoImage(img)

        self.image_label.config(image=img)
        self.image_label.image = img

        for col, var in self.road_vars.items():
            var.set(self.data.iloc[self.index][col])

        for col, var in self.bg_vars.items():
            var.set(self.data.iloc[self.index][col])

        # Update the position label
        self.slider.set(self.index)  # Set slider value to current image index
        self.update_position_label()  # Update position label

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

    def update_position_label(self):
        # Update the position label to show the current position
        self.position_label.config(text=f"Image {self.index + 1} of {len(self.data)}")

    def save_current_scores(self):
        for col, var in self.road_vars.items():
            self.data.loc[self.index, col] = var.get()

        for col, var in self.bg_vars.items():
            self.data.loc[self.index, col] = var.get()

    def save_changes(self):
        self.data.to_csv('image_data.csv', index=False)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageReviewer(root)
    root.mainloop()