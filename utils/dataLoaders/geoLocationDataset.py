import os
import numpy as np
import torch
import sys
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load the coordinates from the cache
cur_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COORDINATES_CACHE = np.load(os.path.join(cur_dir, 'country_cord.npy'), allow_pickle=True).item()

# Define the dataset class
class GeoLocationDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.dataset = ImageFolder(root=root_dir)
        self.datapos = []
        print(self.dataset.class_to_idx)
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

    def get_country_name(self, index):
        return self.dataset.classes[self.dataset.targets[index]]

    def __len__(self):
        return len(self.dataset)


# Debugging to check if the dataset works
if __name__ == "__main__":
    dir_path = r'C:\Users\Muku\OneDrive - Aalborg Universitet\Geo_sets\50k_country_only'
    # Check if the dataclass is correct
    dataset = GeoLocationDataset(root_dir=dir_path, transform=None)

    for i in range(100):
        print(dataset.dataset.targets[i])
        print(dataset.get_country_name(i))