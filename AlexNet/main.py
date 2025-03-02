import torch
from model import AlexNet
from trainer import train_model
from data_loader import get_dataloaders
from config import Config


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # Load data
    train_dataloader, val_dataloader = get_dataloaders()

    # Initialize model
    model = AlexNet()
    model.to(device)

    # Train model and display metrics
    print("Starting training...")
    trained_model, train_losses, val_accuracies = train_model(model, train_dataloader, val_dataloader, num_epochs = Config.EPOCHS, device = None)
    print("Training completed.")
    print(f"Final Training Losses: {train_losses}")
    print(f"Final Test Accuracy: {val_accuracies[-1]:.2f}%")
    torch.save(trained_model.state_dict(), 'alexnet.pth')
    


if __name__ == "__main__":
    main()