import torch
import os
import gc
import torchvision
from tqdm import tqdm
from torchvision import models
from packaging import version
import torch.optim as optim
import sys
from utils.dataLoader import get_dataloaders
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
            self.resnet = models.resnet152(weights="IMAGENET1K_V2")
        else:
            # Load pre-trained ResNet model
            self.resnet = models.resnet152(pretrained=True)

        # Freeze the layers
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Change the final layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Linear(num_features, num_classes)

        # Move the model to the device
        self.resnet = self.resnet.to(self.device)

    def forward(self, x):
        x = self.resnet(x)
        return torch.nn.functional.log_softmax(x, dim=1)


def calculate_accuracy(outputs, labels):
    """Calculate classification accuracy."""
    _, pred = torch.max(outputs, dim=1)
    target = torch.argmax(labels, dim=1)
    correct = pred.eq(target).sum().item()
    return correct / len(pred)


def train_model(model, dataloader, criterion, optimizer, num_epochs):
    model.train()  # set the model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_acc = 0.0
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
            running_acc += calculate_accuracy(outputs, labels) * inputs.size(0)

            progress_bar.set_description(
                f'Epoch {epoch + 1}/{num_epochs} Loss: {loss.item():.4f} Accuracy: {running_acc / (i + 1):.4f}')

            # Delete tensors that are no longer needed
            del inputs, labels, outputs, loss

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_acc / len(progress_bar)
        print(
            f'Epoch {epoch + 1}/{num_epochs} Average Loss: {epoch_loss:.4f} Average Accuracy: {epoch_acc:.4f}')

        os.makedirs('model_checkpoints', exist_ok=True)
        # Save model state after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            'accuracy': epoch_acc,
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

    # Training session
    num_epochs = 50
    learning_rate = 0.05
    dir = r'C:\Users\mikip\Pictures\50k_countryonly'
    num_classes = len(os.listdir(dir))  # amount of countries

    transform = transforms.Compose([
        transforms.Resize((280, 640)),
        transforms.ToTensor(),
    ])

    train_loader = get_dataloaders(dir, transform, 64, num_workers=0, split_set=False)

    model = GeoLocationClassifier(num_classes)

    # Define a loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Define an optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Call the training function
    train_model(model, train_loader, criterion, optimizer, num_epochs)