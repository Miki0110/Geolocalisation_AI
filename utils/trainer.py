import torch
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import sys
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.optim as optim
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.lossFunctions import GeographicalCrossEntropyLoss
from utils.model import ResnetClassifier
from utils.dataLoaders.dataLoader import DataSet
from utils.dataLoaders.geoLocationDataset import GeoLocationDataset


def calculate_accuracy(outputs, labels):
    """Calculate classification accuracy."""
    _, pred = torch.max(outputs, dim=1)
    correct = pred.eq(labels).sum().item()
    return correct / len(pred)


def train_model(model, dataloader, criterion, optimizer, num_epochs, session_name="model", start_epoch=0, scheduler=None):
    """
    Trainer for the model
    Args:
        model (nn.Module): Model used to train on
        dataloader (Dataloader): Dataloader for the training set and test set
        criterion (nn.Module.loss): Loss function
        optimizer (torch.optim): optimizer standard is to just use Adam
        num_epochs (int): Number of epochs
        session_name (str): Name of the session
        start_epoch (int): Start epoch
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler
    """
    # Get Current time
    curr_time = datetime.now().strftime("%H%M%S")
    # Set up a Summarywriter
    writer = SummaryWriter(f'runs/{session_name}_{curr_time}')

    train_loader = dataloader.train_set
    test_loader = dataloader.test_set

    model.train()  # set the model to training mode
    best_acc = 0.0 # Keep track of the best accuracy
    best_loss = 100000000
    for epoch in range(start_epoch, num_epochs):
        running_loss = 0.0
        running_corrects = 0.0
        total_samples = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
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
            running_corrects += torch.sum(preds == labels.data).double()  # Check how many are correct
            total_samples += labels.size(0)

            progress_bar.set_description(
                f'Epoch {epoch + 1}/{num_epochs} Loss: {loss.item():.4f} '
                f'Accuracy: {(running_corrects / total_samples) * 100:.2f}%')

            # Delete tensors that are no longer needed
            del inputs, labels, outputs, loss
            # Free up memory if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Update the learning rate
        if scheduler is not None:
            scheduler.step()

        # Print the average scores
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / total_samples

        # Test the model
        test_loss, test_acc = test_model(model, test_loader, criterion)

        print(f'Epoch {epoch + 1}/{num_epochs} Average Loss: {epoch_loss:.4f}', f"Accuracy: {test_acc*100}%")

        # Write it into the tensorboard
        writer.add_scalar('Loss/Training', epoch_loss, epoch)
        writer.add_scalar('Accuracy/Training', epoch_acc*100, epoch)
        writer.add_scalar('Loss/Test', test_loss, epoch)
        writer.add_scalar('Accuracy/Test', test_acc*100, epoch)

        os.makedirs(f'model_checkpoints', exist_ok=True)
        # Save the model if it is the best one
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_loss,
                'train_accuracy': epoch_acc,
                'test_loss': test_loss,
                'test_accuracy': test_acc
            }, f'model_checkpoints/{session_name}_checkpoint_acc.pth')
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_loss,
                'train_accuracy': epoch_acc,
                'test_loss': test_loss,
                'test_accuracy': test_acc
            }, f'model_checkpoints/{session_name}_checkpoint_loss.pth')

    print('Training complete')
    writer.close()
    return model


def test_model(model, dataloader, criterion):
    model.eval()  # set the model to evaluation mode
    running_loss = 0.0
    running_corrects = 0.0
    total_samples = 0
    with torch.no_grad():  # turn off gradients for the validation, saves memory and computations
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Testing the model")
        for i, (inputs, labels) in progress_bar:
            inputs = inputs.to(model.device)
            labels = labels.to(model.device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).double()
            total_samples += labels.size(0)

            progress_bar.set_description(
                f'Total samples: {total_samples}, '
                f'running corrects: {running_corrects}, '
                f'Accuracy: {(running_corrects / total_samples) * 100:.2f}%')

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects / total_samples
    model.train()  # set the model back to training mode
    return epoch_loss, epoch_acc


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
        raise FileNotFoundError(f"No checkpoint found at {filename}")

    return model, optimizer, start_epoch


if __name__ == '__main__':
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
    session_name = "101_bye_uruguay"
    dir_path = r'C:\Users\Muku\OneDrive - Aalborg Universitet\Geo_sets\50k_country_only'

    # Hyperparameters
    num_classes = len(os.listdir(dir_path))  # amount of countries
    num_epochs = 50
    learning_rate = 0.01
    batch_size = int(64*2)

    resnet_version = 101_2
    dataloader = DataSet(root_dir=dir_path, loader=GeoLocationDataset, transform=transform)
    dataloader.get_dataloaders(batch_size=batch_size, split_set=True)

    # Context model paths
    road_model_name = 'road_notestset.pth'
    bg_model_name = 'background_notestset.pth'
    model = ResnetClassifier(num_classes, road_model_name, bg_model_name, resnet_version=resnet_version)

    # Define a loss function
    criterion = GeographicalCrossEntropyLoss()

    # Define an optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Ask the user if they want to load from a checkpoint
    load_checkpoint_decision = input("Do you want to load from a previous checkpoint? (yes/no): ")
    start_epoch = 0
    # If they do, ask them which epoch
    if (load_checkpoint_decision.lower() == "yes") or (load_checkpoint_decision.lower() == "y"):
        checkpoint_file = f'model_checkpoints/{session_name}_checkpoint.pth'
        # Load the checkpoint
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_file)

    # Call the training function
    train_model(model, dataloader, criterion, optimizer, num_epochs, start_epoch=start_epoch,
                session_name=session_name, scheduler=scheduler)