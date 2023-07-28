import os
import csv
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ContextDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, road=True):
        self.root_dir = root_dir
        self.transform = transform
        self.road = road
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.data_frame = []
        with open(csv_file, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip header
            for row in csv_reader:
                self.data_frame.append(row)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame[idx][1])
        image = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
        """ 
        country = self.data_frame[idx][0]
        latitude = float(self.data_frame[idx][2])
        longitude = float(self.data_frame[idx][3])
        """
        road_labels = [float(i) for i in self.data_frame[idx][4:14]]
        background_labels = [float(i) for i in self.data_frame[idx][14:]]
        if self.transform:
            image = self.transform(image=image)['image'].to(self.device)
        labels = torch.tensor(road_labels).to(self.device) if self.road else torch.tensor(background_labels).to(self.device)
        return image, labels


# Debugging to check if the dataset works
if __name__ == "__main__":
    # Define the transformation
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
        ToTensorV2()
    ])


    # The following code is for testing the context dataset
    csv_file = r'C:\Users\mikip\Documents\Geolocalisation_AI\data_gathering\image_data.csv'
    dir = r'C:\Users\mikip\Documents\Geolocalisation_AI\data_gathering'

    dataset = ContextDataset(csv_file=csv_file, root_dir=dir, transform=transform, road=True)
    data_iter = iter(dataset)
    # Get a batch of data
    for i in range(19):
        items = next(data_iter)
    images, labels = items

    # Print the shapes and device of the images and labels
    print("Images shape:", images.shape)
    print("Images device:", images.device)
    print("Labels shape:", labels.shape)
    print("Labels device:", labels.device)

    # Print the first label
    print("background label:", labels)

    # Data loader test
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    for i, (images, labels) in enumerate(dataloader):
        print("Images shape:", images.shape)
        print("Images device:", images.device)
        print("Labels shape:", labels.shape)
        print("Labels device:", labels.device)
        print("background label:", labels)
