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
from utils.dataLoader import get_dataloaders
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class GeoLocationClassifier(torch.nn.Module):
    """
    GeoLocationClassifier class
    Uses the resnet101_2 model from torchvision
    """
    def __init__(self, num_classes):
        super(GeoLocationClassifier, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Check torchvision version
        if version.parse(torchvision.__version__) >= version.parse('0.13'):
            # Load pre-trained ResNet model
            self.resnet = models.wide_resnet101_2(weights="IMAGENET1K_V2")
        else:
            # Load pre-trained ResNet model
            self.resnet = models.wide_resnet101_2(pretrained=True)

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


def train_model(model, dataloader, criterion, optimizer, num_epochs, start_epoch=0):
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
    writer = SummaryWriter(f'runs/training_{curr_time}')

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
            # Calculate the running accuracy
            _, preds = torch.max(outputs, 1)  # Get the best guess from the output
            running_corrects += torch.sum(preds == labels.data)  # Check how many are correct
            total_samples += labels.size(0)

            progress_bar.set_description(
                f'Epoch {epoch + 1}/{num_epochs} Loss: {loss.item():.4f} '
                f'Accuracy: {(running_corrects.double() / total_samples) * 100:.2f}%')

            # Delete tensors that are no longer needed
            del inputs, labels, outputs, loss
        # Print the average scores
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / total_samples
        print(f'Epoch {epoch + 1}/{num_epochs} Average Loss: {epoch_loss:.4f} '
            f'Average Accuracy: {epoch_acc*100:.2f}%')

        # Write it into the tensorboard
        writer.add_scalar('Training Loss', epoch_loss, epoch)
        writer.add_scalar('Training Accuracy', epoch_acc*100, epoch)

        os.makedirs('model_checkpoints', exist_ok=True)
        # Save model state after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            'accuracy': epoch_acc,
        }, f'model_checkpoints/checkpoint_{epoch}.pth')
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

    # Training session
    num_epochs = 50
    learning_rate = 0.01
    dir = r'C:\Users\mikip\Pictures\50k_countryonly'
    num_classes = len(os.listdir(dir))  # amount of countries

    # Augmentations applied to make up for street view changes
    transform = A.Compose([
        A.Resize(200, 440),
        A.HorizontalFlip(p=0.5),  # Horizontal flip
        A.RandomBrightnessContrast(p=0.2),  # Random brightness and contrast
        A.Rotate(limit=20, p=0.5),  # Random rotation
        A.CoarseDropout(max_holes=4, max_height=40, max_width=40, p=0.5),  # Change the image by cutting out parts
        A.GaussNoise(p=0.5),  # Add gaussian noise
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image (This is for the pretrained model)
        ToTensorV2(),
    ])

    train_loader = get_dataloaders(dir, transform, 64, num_workers=0, split_set=False)

    model = GeoLocationClassifier(num_classes)

    # Define a loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Define an optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Ask the user if they want to load from a checkpoint
    load_checkpoint_decision = input("Do you want to load from a previous checkpoint? (yes/no): ")
    start_epoch = 0
    # If they do, ask them which epoch
    if (load_checkpoint_decision.lower() == "yes") or (load_checkpoint_decision.lower() == "y"):
        epoch_number = input("Please enter the epoch of the checkpoint: ")
        checkpoint_file = f'model_checkpoints/checkpoint_{epoch_number}.pth'
        # Load the checkpoint
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_file)

    # Call the training function
    train_model(model, train_loader, criterion, optimizer, num_epochs, start_epoch=start_epoch)