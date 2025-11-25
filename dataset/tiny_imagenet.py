from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATASET_MEAN = [0.485, 0.456, 0.406]
DATASET_STD = [0.229, 0.224, 0.225]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=DATASET_MEAN,
                         std=DATASET_STD)
])

VAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=DATASET_MEAN,
                         std=DATASET_STD)
])


class TinyImageNet():
    
    def __init__(self, path, transform = True):
        self.train_set = datasets.ImageFolder(path + '/train', TRAIN_TRANSFORM)
        self.val_set = datasets.ImageFolder(path + '/val', VAL_TRANSFORM)

    def get_dataloaders(self, batch_size = 128, num_workers = 4):
        trainloader = DataLoader(self.train_set, batch_size = batch_size, shuffle = True, num_workers = num_workers)
        valloader = DataLoader(self.val_set, batch_size = batch_size, shuffle = False, num_workers = num_workers)

        return trainloader, valloader

