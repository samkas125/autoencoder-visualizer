# autoencoder/utils.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import os

def get_mnist_dataloaders(batch_size=64):
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def show_reconstructions(model, dataloader, device, save_path="outputs/reconstructions/vis.png", num_images=6):
    model.eval()
    images, _ = next(iter(dataloader))
    images = images.to(device)

    with torch.no_grad():
        recon, _ = model(images)

    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
    for i in range(num_images):
        axes[0, i].imshow(images[i][0].cpu(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(recon[i][0].cpu(), cmap='gray')
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel("Original", fontsize=14)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=14)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_latent_space(model, dataloader, device, save_path="outputs/latent/tsne.png", num_samples=1000):
    model.eval()
    latent_vectors = []
    labels = []

    count = 0
    for images, targets in dataloader:
        images = images.to(device)
        with torch.no_grad():
            _, z = model(images)
        latent_vectors.append(z.cpu().numpy())
        labels.append(targets.numpy())
        count += len(images)
        if count >= num_samples:
            break

    latent_vectors = np.concatenate(latent_vectors, axis=0)[:num_samples]
    labels = np.concatenate(labels, axis=0)[:num_samples]

    tsne = TSNE(n_components=2, random_state=42)
    z_2d = tsne.fit_transform(latent_vectors)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap='tab10', s=15)
    plt.colorbar(scatter, ticks=range(10), label="Digit Label")
    plt.title("t-SNE of Latent Space")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
