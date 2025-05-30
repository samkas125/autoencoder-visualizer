# visualize.py
import torch
from autoencoder.model import Autoencoder
from autoencoder.utils import get_mnist_dataloaders, show_reconstructions, plot_latent_space

def main():
    # Hyperparameters
    latent_dim = 32
    model_path = "models/autoencoder_32.pth"

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test data
    _, test_loader = get_mnist_dataloaders(batch_size=128)

    # Load model
    model = Autoencoder(latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")

    # Run visualizations
    show_reconstructions(model, test_loader, device)
    print("Reconstruction visualization saved.")

    plot_latent_space(model, test_loader, device)
    print("Latent space visualization saved.")

if __name__ == "__main__":
    main()
