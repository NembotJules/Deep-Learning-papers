import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from config import Config


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataloaders(): 
    """
    Load and return train and test dataloaders...
    """
    # Training transforms (with augmentation for 227x227)
    train_transform = transforms.Compose([
    transforms.Resize(256),  
    transforms.RandomCrop(227),  
    transforms.RandomHorizontalFlip(p=0.5),  # 50% chance of flipping
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),  # Lighting changes
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

   # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
    transforms.Resize(256),  
    transforms.CenterCrop(227),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])
    train_dataset = datasets.ImageFolder(root='tiny-imagenet-200/train', transform = train_transform)
    valid_dataset = datasets.ImageFolder(root = 'tiny-imagenet-200/val', transform= val_transform)

    train_dataloader = DataLoader(train_dataset, batch_size = Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, shuffle = True)
    val_dataloader = DataLoader(valid_dataset, batch_size = Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, shuffle = True)

    return train_dataloader, val_dataloader