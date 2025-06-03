import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from autoencoder.model import Autoencoder

# Load the autoencoder model
def load_model(latent_dim=32, model_path="models/autoencoder_32.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

# Generate synthetic images
def generate_synthetic_images(input_image_path, latent_dim=32, model_path="models/autoencoder_32.pth", num_images=10):
    model, device = load_model(latent_dim, model_path)

    # preprocess
    image = Image.open(input_image_path).convert("L").resize((28, 28))
    transform = transforms.ToTensor()
    tensor_img = transform(image).unsqueeze(0).to(device)

    # Encode the image to latent space
    with torch.no_grad():
        z = model.encoder(tensor_img)

    # sample using stdev of latent vector
    synthetic_images = []
    for _ in range(num_images):
        std_dev = torch.std(z) * 0.2
        noise = torch.normal(mean=0, std=std_dev, size=z.size(), device=z.device)
        z_noisy = z + noise # Add noise
        with torch.no_grad():
            recon = model.decoder(z_noisy)
        synthetic_images.append(recon.squeeze(0).cpu().numpy())

    return synthetic_images

def save_synthetic_images(synthetic_images, output_dir="outputs/synthetic/"):
    import os
    os.makedirs(output_dir, exist_ok=True)
    for i, img_array in enumerate(synthetic_images):
        img = Image.fromarray((img_array[0] * 255).astype(np.uint8), mode="L")
        img.save(os.path.join(output_dir, f"synthetic_{i}.png"))

if __name__ == "__main__":
    input_image_path = "path/to/image"  # Add image path
    synthetic_images = generate_synthetic_images(input_image_path=input_image_path, 
                                                 latent_dim=32, 
                                                 model_path="models/autoencoder_32.pth",
                                                 num_images=10)
    save_synthetic_images(synthetic_images)
    print(f"Generated {len(synthetic_images)} synthetic images.")