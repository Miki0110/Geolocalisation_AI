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
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

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
class EuclideanLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, labels):
        return ((preds - labels)**2).sum(dim=-1).sqrt().mean()


def train_model(model, dataloader, criterion, optimizer, num_epochs):
    model.train()  # set the model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        total_distance_error = [0.0, 0]  # Total error in km for this epoch
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
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            # Calculate Haversine distance error for this batch
            outputs_cpu = outputs.detach().cpu().numpy()
            labels_cpu = labels.detach().cpu().numpy()

            for pred, true in zip(outputs_cpu, labels_cpu):
                total_distance_error[0] += haversine_distance(pred[0], pred[1], true[0], true[1])
                total_distance_error[1] += 1

            progress_bar.set_description(f'Epoch {epoch + 1}/{num_epochs} Loss: {loss.item():.4f} Error: {total_distance_error[0]/total_distance_error[1]:.4f} km')

        epoch_loss = running_loss / len(dataloader.dataset)
        average_distance_error = total_distance_error[0] / total_distance_error[1]
        print(
            f'Epoch {epoch + 1}/{num_epochs} Average Loss: {epoch_loss:.4f}, Average Error: {average_distance_error:.4f} km')

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
    learning_rate = 0.001
    dir = r'C:\Users\Muku\OneDrive - Aalborg Universitet\Geo_sets\50k_country_only'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
    ])

    train_loader, test_loader = get_dataloaders(dir, transform, 64, num_workers=2, split_set=True)

    model = GeoLocationClassifier(num_classes)

    # Define a loss function
    criterion = EuclideanLoss()

    # Define an optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Call the training function
    train_model(model, train_loader, criterion, optimizer, num_epochs)

    # Call the testing function
    test_model(model, test_loader, criterion)