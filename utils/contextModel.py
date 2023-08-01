import torch
from datetime import datetime
import os
import gc
import torchvision
from tqdm import tqdm
from torchvision import models
from packaging import version
import torch.optim as optim
import sys
from utils.dataLoaders.dataLoader import DataSet
from utils.dataLoaders.contextDataset import ContextDataset
from utils.dataLoaders.geoLocationDataset import GeoLocationDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ContextModel(torch.nn.Module):
    """
    Context model for the geolocation AI
    """
    def __init__(self, num_classes, resnet_version=101_2):
        super(ContextModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Check torchvision version
        if version.parse(torchvision.__version__) >= version.parse('0.13'):
            # Load pre-trained ResNet model
            if resnet_version == 101_2:
                self.resnet = models.wide_resnet101_2(weights="IMAGENET1K_V2")
            elif resnet_version == 50_2:
                self.resnet = models.wide_resnet50_2(weights="IMAGENET1K_V2")
            elif resnet_version == 152:
                self.resnet = models.resnet152(weights="IMAGENET1K_V2")
            elif resnet_version == 101:
                self.resnet = models.resnet101(weights="IMAGENET1K_V2")
            elif resnet_version == 50:
                self.resnet = models.resnet50(weights="IMAGENET1K_V2")
            else:
                raise ValueError("Invalid resnet version")
        else:
            # Load pre-trained ResNet model
            if resnet_version == 101_2:
                self.resnet = models.wide_resnet101_2(pretrained=True)
            elif resnet_version == 50_2:
                self.resnet = models.wide_resnet50_2(pretrained=True)
            elif resnet_version == 152:
                self.resnet = models.resnet152(pretrained=True)
            elif resnet_version == 101:
                self.resnet = models.resnet101(pretrained=True)
            elif resnet_version == 50:
                self.resnet = models.resnet50(pretrained=True)
            else:
                raise ValueError("Invalid resnet version")

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
        return x


def calculate_accuracy(outputs, labels):
    """Calculate classification accuracy."""
    _, pred = torch.max(outputs, dim=1)
    correct = pred.eq(labels).sum().item()
    return correct / len(pred)


def train_model(model, dataloader, criterion, optimizer, num_epochs, session_name="model", start_epoch=0, road=None):
    """
    Trainer for the model
    Args:
        model (nn.Module): Model used to train on
        dataloader (torch.data.dataloader): Dataloader for the training set
        criterion (nn.Module.loss): Loss function
        optimizer (torch.optim): optimizer standard is to just use Adam
        num_epochs (int): Number of epochs
    """
    # Set up a Summarywriter
    curr_time = datetime.now().strftime("%H%M%S")
    writer = SummaryWriter(f'runs/{session_name}_{curr_time}')

    model.train()  # set the model to training mode
    for epoch in range(start_epoch, num_epochs):
        running_loss = 0.0
        running_corrects = 0.0
        total_samples = 0
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, (inputs, labels) in progress_bar:
            # Set the tensor to the right device
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

            # Calculate the running loss
            running_loss += loss.item() * inputs.size(0)
            if road is None:
                # Calculate the running accuracy
                _, preds = torch.max(outputs, 1)  # Get the best guess from the output
                running_corrects += torch.sum(preds == labels.data).double()  # Check how many are correct
            else:
                preds = torch.sigmoid(outputs) > 0.8
                # Exact match for the accuracy
                running_corrects += (preds == labels).all(dim=1).float().sum().item()  # Add it to your running total
            total_samples += labels.size(0)  # Update total number of samples
            progress_bar.set_description(
                f'Epoch {epoch + 1}/{num_epochs} Loss: {loss.item():.4f} '
                f'Accuracy: {(running_corrects / total_samples) * 100:.2f}%')

            # Delete tensors that are no longer needed
            del inputs, labels, outputs, loss
        # Print the average scores
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects / total_samples
        print(f'Epoch {epoch + 1}/{num_epochs} Average Loss: {epoch_loss:.4f} ')

        # Write it into the tensorboard
        writer.add_scalar('Training Loss', epoch_loss, epoch)
        writer.add_scalar('Training Accuracy', epoch_acc*100, epoch)

        os.makedirs(f'model_checkpoints', exist_ok=True)
        # Save model state after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            'accuracy': epoch_acc,
        }, f'model_checkpoints/{session_name}_checkpoint_{epoch}.pth')
        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()

    print('Training complete')
    writer.close()
    return model


def load_checkpoint(model, optimizer, filename):
    """
    For Loading a checkpoint
    """
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch


# Debugging to check if the model works and cuda is available
if __name__ == "__main__":
    # Check for cuda
    print("Cuda is", "available" if torch.cuda.is_available() else "not available")

    # Augmentations applied to make up for street view changes
    transform = A.Compose([
        A.Resize(200, 440),
        A.HorizontalFlip(p=0.5),  # Horizontal flip
        A.RandomBrightnessContrast(p=0.2),  # Random brightness and contrast
        A.Rotate(limit=20, p=0.5),  # Random rotation
        A.CoarseDropout(max_holes=4, max_height=40, max_width=40, p=0.5),  # Change the image by cutting out parts
        A.GaussNoise(p=0.5),  # Add gaussian noise
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
        ToTensorV2(),
    ])
    session_name = "background"
    dir_path = r'C:\Users\Muku\Documents\Geolocalisation_AI\data_gathering'
    road = False

    # Hyperparameters
    num_classes = 10
    num_epochs = 50
    learning_rate = 0.1

    resnet_version = 152
    dataloader = DataSet(root_dir=dir_path, loader=ContextDataset, transform=transform)
    dataloader.get_dataloaders(64, road=road)
    train_loader = dataloader.train_set

    model = ContextModel(num_classes, resnet_version=resnet_version)

    # Define a loss function
    criterion = torch.nn.BCEWithLogitsLoss()

    # Define an optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Ask the user if they want to load from a checkpoint
    load_checkpoint_decision = input("Do you want to load from a previous checkpoint? (yes/no): ")
    start_epoch = 0
    # If they do, ask them which epoch
    if (load_checkpoint_decision.lower() == "yes") or (load_checkpoint_decision.lower() == "y"):
        epoch_number = input("Please enter the epoch of the checkpoint: ")
        checkpoint_file = f'model_checkpoints/{session_name}_checkpoint_{epoch_number}.pth'
        # Load the checkpoint
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_file)

    # Call the training function
    train_model(model, train_loader, criterion, optimizer, num_epochs, start_epoch=start_epoch, session_name=session_name, road=road)