import os
import numpy as np
import torch
import sys
from geopy.geocoders import Nominatim
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import albumentations as A

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_country_coordinates(country):
    geolocator = Nominatim(user_agent="country_converter")
    location = geolocator.geocode(country, exactly_one=True)

    if location is None:
        return None

    latitude = location.latitude
    longitude = location.longitude
    return latitude, longitude

# Load the coordinates from the cache
cur_dir = os.path.dirname(os.path.abspath(__file__))
COORDINATES_CACHE = np.load(os.path.join(cur_dir, 'country_cord.npy'), allow_pickle=True).item()
# Function to parse the folder name and get the coordinates
def parse_folder_name(folder_name):
    # If the coordinates are already in the cache, use them
    if folder_name in COORDINATES_CACHE:
        return COORDINATES_CACHE[folder_name]

    # Otherwise, fetch the coordinates and store them in the cache
    coords = get_country_coordinates(folder_name)
    if coords is None:
        raise ValueError("Invalid folder name: " + folder_name)

    COORDINATES_CACHE[folder_name] = coords
    return coords


# Define the dataset class
class GeoLocationDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.dataset = ImageFolder(root=root_dir)
        self.datapos = []
        for data_class in self.dataset.classes:
            self.datapos.append(COORDINATES_CACHE[data_class])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __getitem__(self, index):
        image, _ = self.dataset[index]  # Ignore the original label
        # For getting the name of the folder
        # folder_name = self.dataset.classes[self.dataset.targets[index]]
        # Create the label
        country = self.dataset.targets[index]
        country = torch.tensor(country)
        # Apply the transformation
        if self.transform is not None:
            image = self.transform(image=np.array(image))['image']
        return image.to(self.device), country.to(self.device)

    def __len__(self):
        return len(self.dataset)

# Function to get the data loaders
def get_dataloaders(root_dir, transform, batch_size, num_workers = 1, split_set=False):
    dataset = GeoLocationDataset(root_dir=root_dir, transform=transform)

    if split_set:
        # Determine the lengths of your splits (here, 80% training, 20% testing)
        train_length = int(0.8 * len(dataset))
        test_length = len(dataset) - train_length

        # Create the splits
        train_dataset, test_dataset = random_split(dataset, [train_length, test_length])

        # Create the data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_loader, test_loader
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)



# Debugging to check if the dataset works
if __name__ == "__main__":
    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
    ])
    dir = r'C:\Users\mikip\Pictures\50k_countryonly'
    # Create datasets
    datasets = get_dataloaders(root_dir=dir, transform=transform, batch_size=12, num_workers= 0, split_set=True)

    for data_loader in datasets:
        data_iter = iter(data_loader)

        # Get a batch of data
        images, labels = next(data_iter)

        # Print the shapes and device of the images and labels
        print("Images shape:", images.shape)
        print("Images device:", images.device)
        print("Labels shape:", labels.shape)
        print("Labels device:", labels.device)

        # Print the first label to check if the coordinates make sense
        print("First label:", labels[0])