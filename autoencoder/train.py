# autoencoder/train.py
import torch
from torch import nn, optim
from model import Autoencoder
from utils import get_mnist_dataloaders

def train_autoencoder(epochs=10, batch_size=64, latent_dim=32, lr=1e-3, save_model=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_mnist_dataloaders(batch_size)
    model = Autoencoder(latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            recon, _ = model(imgs)
            loss = criterion(recon, imgs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f}")
    
    if save_model:
        torch.save(model.state_dict(), "autoencoder_64.pth")
        print("Model saved to autoencoder.pth")
    

if __name__ == "__main__":
    train_autoencoder(latent_dim=32)
