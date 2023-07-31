from torch.utils.data import DataLoader, random_split

class DataSet:
    def __init__(self, root_dir, loader, transform=None):
        self.transform = transform
        self.root_dir = root_dir
        self.loader = loader
        self.test_set = None
        self.train_set = None

    # Function to get the data loaders
    def get_dataloaders(self, batch_size, split_set=False, num_workers=0, road=None):
        csv_file = f'{self.root_dir}\\image_data.csv'
        if road is not None:
            dataset = self.loader(csv_file=csv_file, root_dir=self.root_dir, transform=self.transform, road=road)
        else:
            dataset = self.loader(root_dir=self.root_dir, transform=self.transform)

        if split_set:
            # Determine the lengths of your splits (here, 80% training, 20% testing)
            train_length = int(0.8 * len(dataset))
            test_length = len(dataset) - train_length

            # Create the splits
            train_dataset, test_dataset = random_split(dataset, [train_length, test_length])

            # Create the data loaders
            self.train_set = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            self.test_set = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        else:
            self.train_set = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    def __len__(self):
        return len(self.test_set) + len(self.train_set)