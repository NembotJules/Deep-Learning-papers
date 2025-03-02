import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from config import Config


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataloaders(): 
    """
    Load and return train and test dataloaders...
    """
    transform = transforms.Compose([
        transforms.Resize(Config.INPUT_SIZE), 
        transforms.ToTensor(), 

    ])

    train_dataset = datasets.ImageFolder(root='tiny-imagenet-200/train', transform = transform)
    valid_dataset = datasets.ImageFolder(root = 'tiny-imagenet-200/val', transform= transform)

    train_dataloader = DataLoader(train_dataset, batch_size = Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, shuffle = True)
    val_dataloader = DataLoader(valid_dataset, batch_size = Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, shuffle = True)

    return train_dataloader, val_dataloader