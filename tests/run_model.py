import sys
import os
import cv2
import numpy as np
import torch
import random
from torchvision import transforms as T
# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model import GeoLocationClassifier

### INSERT IMAGE FOLDER PATH HERE ###
image_path = r"C:\Users\mikip\Pictures\50k_countryonly"
NUM_TESTS = 50


def get_all_images(root):
    images = []
    for subdir, dirs, files in os.walk(root):
        for file in files:
            # Check if the file is an image (you can add more types if needed)
            if file.endswith(('.png', '.jpg', '.jpeg')):
                images.append(os.path.join(subdir, file))
    return images


# Get the paths to the model folder
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
npy_file_path = os.path.join(parent_dir, 'utils', 'country_cord.npy')

# Load the country coordinates
COORDINATES_CACHE = np.load(npy_file_path, allow_pickle=True).item()
print(COORDINATES_CACHE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = GeoLocationClassifier(num_classes=len(COORDINATES_CACHE)).to(device)

# Load the saved model weights
model_folder = os.path.join(parent_dir, 'utils', 'model_checkpoints')
checkpoint = torch.load(os.path.join(model_folder, "checkpoint_49.pth"))  # path to the saved model
model.load_state_dict(checkpoint['model_state_dict'])

# Set the model to evaluation mode
model.eval()

# Transform for the images loaded
transform = T.Compose([
    T.ToPILImage(),  # because the image is a numpy array
    T.Resize((170, 400)),
    T.ToTensor(),
])

# Get all the images in the folder
all_images = get_all_images(image_path)

# Now you can use the model to make predictions
with torch.no_grad():
    for i in range(NUM_TESTS):
        # Load the image
        image_path = random.sample(all_images, 1)[0]
        img = cv2.imread(image_path)
        pil_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Apply the image to the model
        inputs = transform(pil_img).unsqueeze(0).to(device)

        temp_t = T.ToPILImage()
        torch_vers = temp_t(transform(img)).show()

        outputs = model(inputs)
        # Apply softmax
        outputs = torch.nn.functional.softmax(outputs, dim=1)
        # Pick the 5 largest values
        _, preds = torch.topk(outputs, 5, dim=1)

        # Display the image and the prediction
        cv2.imshow("Image", img)
        for i, pred in enumerate(preds[0]):
            print(f"Prediction {i + 1}: {list(COORDINATES_CACHE.keys())[pred.item()]}")
            print(f"Probability: {outputs[0][pred.item()].item()}")
        print("Actual:", os.path.basename(os.path.dirname(image_path)))
        cv2.waitKey(0)
