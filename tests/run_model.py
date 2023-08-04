import sys
import os
import numpy as np
import torch
import random
# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model import ResnetClassifier
import tkinter as tk
from PIL import Image, ImageTk
import albumentations as A
from albumentations.pytorch import ToTensorV2

### INSERT IMAGE FOLDER PATH HERE ###
image_path = r"C:\Users\Muku\OneDrive - Aalborg Universitet\Geo_sets\50k_country_only"
NUM_TESTS = 50

class ImagesIterator:
    def __init__(self, root, model, COORDINATES):
        self.root = root
        self.model = model
        self.COORDINATES = list(COORDINATES.keys())
        self.window = tk.Tk()
        self.images = self._get_all_images()
        if len(self.images) == 0:
            raise ValueError("No images found in the directory")
        self.index = 0

    def current(self):
        return self.images[self.index]

    def next_image(self):
        self.index = (self.index + 1) % len(self.images)
        self.predict_and_display()

    def prev_image(self):
        self.index = (self.index - 1) % len(self.images)
        self.predict_and_display()

    def predict_and_display(self):
        model = self.model
        window = self.window
        # Clear the window
        for widget in window.winfo_children():
            widget.destroy()

        # Load the image
        raw_image = Image.open(self.current())

        # Convert the PIL Image to a NumPy array
        np_image = np.array(raw_image)

        # Get actual country
        actual_country = os.path.basename(os.path.dirname(self.current()))

        # Transformations to apply before feeding the image to the model
        transform = A.Compose([
            A.Resize(200, 440),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
            ToTensorV2(),
        ])

        # Apply transformations and add batch dimension
        image = transform(image=np_image)['image'].unsqueeze(0)
        image = image.to(model.device)

        # Make predictions
        with torch.no_grad():
            output = model(image)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            probs, preds = torch.topk(probabilities, 5, dim=1)

        # Display the image
        window.image = ImageTk.PhotoImage(raw_image)
        tk.Label(window, image=window.image).pack()

        # Display the actual country
        tk.Label(window, text=f'Actual country: {actual_country}').pack()

        # Display the predictions
        for i in range(5):
            tk.Label(window, text=f'Country {self.COORDINATES[preds[0,i].item()]}: {probs[0,i].item() * 100:.2f}%').pack()

        # Next and Previous buttons
        tk.Button(window, text="Next", command=lambda: self.next_image()).pack(side=tk.RIGHT)
        tk.Button(window, text="Previous", command=lambda: self.prev_image()).pack(side=tk.LEFT)

    def _get_all_images(self):
        images = []
        for subdir, dirs, files in os.walk(self.root):
            for file in files:
                # Check if the file is an image (you can add more types if needed)
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    images.append(os.path.join(subdir, file))
        random.shuffle(images)
        return images


if __name__ == "__main__":
    # Get the paths to the model folder
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    npy_file_path = os.path.join(parent_dir, 'utils', 'country_cord.npy')

    # Load the country coordinates
    COORDINATES_CACHE = np.load(npy_file_path, allow_pickle=True).item()
    print(COORDINATES_CACHE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = ResnetClassifier(len(COORDINATES_CACHE), 'road_notestset.pth', 'background_notestset.pth').to(device)

    # Load the saved model weights
    model_folder = os.path.join(parent_dir, 'utils', 'model_checkpoints')
    checkpoint = torch.load(
        os.path.join(model_folder, "context_resnet101_2_with_dropout"))  # path to the saved model
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set the model to evaluation mode
    model.eval()

    # Create the iterator
    iterator = ImagesIterator(image_path, model, COORDINATES_CACHE)

    # Display the first image
    iterator.predict_and_display()

    # Start the GUI
    iterator.window.mainloop()


