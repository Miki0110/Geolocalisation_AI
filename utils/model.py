import torch
import os
import gc
import torchvision
from tqdm import tqdm
from torchvision import models
from packaging import version
import torch.optim as optim
from utils.dataLoader import get_dataloaders
from torchvision import transforms
import numpy as np


# Function calculates the distance in km between two points on Earth given their latitudes and longitudes
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # Radius of the Earth in km
    lat1_rad = torch.deg2rad(lat1)
    lon1_rad = torch.deg2rad(lon1)
    lat2_rad = torch.deg2rad(lat2)
    lon2_rad = torch.deg2rad(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.arctan2(torch.sqrt(a), torch.sqrt(1 - a))

    distance = R * c
    return distance

class GeoLocationClassifier(torch.nn.Module):
    """
    GeoLocationClassifier class
    Uses the resnet50 model from torchvision
    """
    def __init__(self, num_classes):
        super(GeoLocationClassifier, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Check torchvision version
        if version.parse(torchvision.__version__) >= version.parse('0.13'):
            # Load pre-trained ResNet model
            self.resnet = models.resnet50(weights="IMAGENET1K_V2")
        else:
            # Load pre-trained ResNet model
            self.resnet = models.resnet50(pretrained=True)

        # Freeze the layers
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Change the final layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Linear(num_features, num_classes)

        # Move the model to the device
        self.resnet = self.resnet.to(self.device)

    def forward(self, x):
        return self.resnet(x)


# Loss function for the model
class HaversineLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, labels):
        lat1, lon1 = preds[:, 0], preds[:, 1]
        lat2, lon2 = labels[:, 0], labels[:, 1]
        return haversine_distance(lat1, lon1, lat2, lon2).mean()


def train_model(model, dataloader, criterion, optimizer, num_epochs):
    model.train()  # set the model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, (inputs, labels) in progress_bar:
            inputs = inputs.to(model.device)
            labels = labels.to(model.device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # backward + optimize
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            progress_bar.set_description(f'Epoch {epoch + 1}/{num_epochs} Loss: {loss.item():.4f} km')

            # Delete tensors that are no longer needed
            del inputs, labels, outputs, loss

        epoch_loss = running_loss / len(dataloader.dataset)
        print(
            f'Epoch {epoch + 1}/{num_epochs} Average Loss: {epoch_loss:.4f} km')

        os.makedirs('model_checkpoints', exist_ok=True)
        # Save model state after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, f'model_checkpoints/checkpoint_{epoch}.pth')

        torch.cuda.empty_cache()
        gc.collect()

    print('Training complete')
    return model


def test_model(model, dataloader, criterion):
    model.eval()  # set the model to evaluation mode
    running_loss = 0.0
    with torch.no_grad():
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, (inputs, labels) in progress_bar:
            inputs = inputs.to(model.device)
            labels = labels.to(model.device)

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            progress_bar.set_description(f'Test Loss: {loss.item():.4f}')
        average_loss = running_loss / len(dataloader.dataset)
        print(f'Average Test Loss: {average_loss:.4f}')


# Debugging to check if the model works and cuda is available
if __name__ == "__main__":
    # Check for cuda
    print("Cuda is", "available" if torch.cuda.is_available() else "not available")

    num_classes = 2  # Longitude and Latitude
    num_epochs = 10
    learning_rate = 0.01
    dir = r'C:\Users\Muku\OneDrive - Aalborg Universitet\Geo_sets\50k_country_only'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
    ])

    train_loader, test_loader = get_dataloaders(dir, transform, 128, num_workers=2, split_set=True)

    model = GeoLocationClassifier(num_classes)

    # Define a loss function
    criterion = HaversineLoss()

    # Define an optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Call the training function
    train_model(model, train_loader, criterion, optimizer, num_epochs)

    # Call the testing function
    test_model(model, test_loader, criterion)