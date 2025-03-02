import torch
from torch import nn
import torch.optim as optim
from config import Config


def train_model(model, train_loader, val_loader, num_epochs = Config.EPOCHS, device = None): 

    if device is None: 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=Config.lr, momentum = 0.9)

    train_losses = []
    val_accuracies = []

    for epoch in range(Config.EPOCHS): 
        model.train()
        running_loss = 0.0
        for images, labels in train_loader: 
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch +1} / {num_epochs}], Loss: {epoch_loss: .4f}")


        #Evaluate on the validations set...
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad(): 
            for images, labels in val_loader: 
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)
        print(f"Accuracy on test set: {val_accuracy:.2f}%")

    return model, train_losses, val_accuracies