from torch import nn
import os
import numpy as np
import torch

class GeographicalCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(GeographicalCrossEntropyLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight, size_average, ignore_index, reduce, reduction)
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        npy_file_path = os.path.join(parent_dir, 'utils', 'country_cord.npy')
        map_dictionary = np.load(npy_file_path, allow_pickle=True).item()
        # Convert the dictionary to a tensor
        self.country_coordinates = torch.tensor(list(map_dictionary.values()))

    def forward(self, input, target):
        # Compute standard cross-entropy loss
        ce_loss = self.cross_entropy(input, target)

        # Compute geographical loss
        geo_loss = self.compute_geographical_loss(input, target)

        # Combine the two losses in some way (e.g., weighted sum)
        total_loss = ce_loss + geo_loss
        return total_loss

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance in kilometers between two points
        on the earth (specified in decimal degrees)
        """
        # Convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(torch.deg2rad, [lon1, lat1, lon2, lat2])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
        c = 2 * torch.asin(torch.sqrt(a))
        r = 6371  # Radius of earth in kilometers
        return c * r

    def compute_geographical_loss(self, input, target):
        # Get the maximum probability prediction for each instance in the batch
        _, preds = torch.max(input, 1)
        preds = preds.detach().cpu().numpy()  # Detach and convert to numpy
        target = target.detach().cpu().numpy()  # Convert target to numpy

        # Calculate the Haversine distance for each instance in the batch
        distances = []
        for pred, true in zip(preds, target):
            lat1, lon1 = self.country_coordinates[pred]
            lat2, lon2 = self.country_coordinates[true]
            distance = self.haversine_distance(lat1, lon1, lat2, lon2)
            distances.append(distance)

        # Convert distances back to a tensor
        distances = torch.tensor(distances)

        # to get a single value representing the geographical loss for the batch
        geo_loss = torch.mean(distances) / 500  # Weighted by 500 to make it comparable to CE loss
        return geo_loss

if __name__ == "__main__":
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    npy_file_path = os.path.join(parent_dir, 'utils', 'country_cord.npy')
    map_dictionary = np.load(npy_file_path, allow_pickle=True).item()
    #print(map_dictionary)
    # Test the loss function
    loss_fn = GeographicalCrossEntropyLoss()
    #print(loss_fn.country_coordinates)

    # Create some dummy data
    input = torch.randn(1, len(map_dictionary))  # logits for a batch of 10 instances
    target = torch.randint(0, len(map_dictionary), (1,))  # true class indices for each instance in the batch

    # Print the countries
    print('Input country:', list(map_dictionary.keys())[torch.argmax(input)])
    print('Target country:', list(map_dictionary.keys())[target])

    # Compute the loss
    loss = loss_fn(input, target)
