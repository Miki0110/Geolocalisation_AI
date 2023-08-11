import torch
import os
import torchvision
from torchvision import models
from packaging import version
from utils.contextModel import ContextModel

class ResnetClassifier(torch.nn.Module):
    """
    GeoLocationClassifier class
    Uses the resnet101_2 model from torchvision
    """
    def __init__(self, num_classes, road_model_name, bg_model_name, resnet_version=101_2):
        super(ResnetClassifier, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_folder = os.path.join(parent_dir, 'utils', 'model_checkpoints')

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

        # Get the feature length
        num_features = self.resnet.fc.in_features

        # Load road and background models
        bg_checkpoint = torch.load(os.path.join(model_folder, bg_model_name))
        road_checkpoint = torch.load(os.path.join(model_folder, road_model_name))

        # Get the models
        self.bg_model = ContextModel(10, resnet_version=bg_checkpoint['model_version'])
        self.road_model = ContextModel(10, resnet_version=road_checkpoint['model_version'])

        self.bg_model.load_state_dict(bg_checkpoint['model_state_dict'])
        self.road_model.load_state_dict(road_checkpoint['model_state_dict'])

        # Set the models to evaluation mode
        for param in self.bg_model.parameters():
            param.requires_grad = False
        for param in self.road_model.parameters():
            param.requires_grad = False

        # Introduce dropout
        self.dropout = torch.nn.Dropout(0.3)

        # Modify the final layer to take into account the features from the other two models
        con_features = self.road_model.resnet.fc.out_features
        con_features += self.bg_model.resnet.fc.out_features

        # Remove the last fully connected layer
        modules = list(self.resnet.children())[:-1]

        # Define the resnet_conv to include all layers up to the fc layer
        self.resnet_conv = torch.nn.Sequential(*modules).to(self.device)

        # Add the new fully connected layers
        self.resnet.fc = torch.nn.Sequential(
            self.dropout,
            torch.nn.Linear(num_features, 256),
        ).to(self.device)
        self.context_fc = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(con_features+256, num_classes),
        ).to(self.device)

        # Move the model to the device
        self.resnet = self.resnet.to(self.device)

    def forward(self, x):
        # Forward through the context models
        road_features = self.road_model(x)
        bg_features = self.bg_model(x)

        # Forward through the resnet conv layers
        x = self.resnet_conv(x)
        # Flatten the features
        x = torch.flatten(x, 1)
        # Apply the bottleneck layer
        x = self.resnet.fc(x)

        # Concatenate the features
        x = torch.cat((road_features, bg_features, x), dim=1)

        # Forward through the final layer
        x = self.context_fc(x)
        return x


# Debugging to check if the model works and cuda is available
if __name__ == "__main__":
    # Check for cuda
    print("Cuda is", "available" if torch.cuda.is_available() else "not available")

    # Create a random tensor
    x = torch.randn(1, 3, 200, 420)
    x = x.to("cuda")

    # Create the model
    model = ResnetClassifier(10, "road_notestset.pth", "background_notestset.pth")
    # Apply the model
    output = model(x)
    print(output)


