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



# Debugging to check if the dataset works
if __name__ == "__main__":
    # Define the transformation
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
        ToTensorV2()
    ])
    dir = r'C:\Users\mikip\Pictures\50k_countryonly'
    # Create datasets
    datasets = get_dataloaders(root_dir=dir, transform=transform, batch_size=12, num_workers=0, split_set=True)

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