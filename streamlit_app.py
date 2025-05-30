import streamlit as st
import torch
from autoencoder.model import Autoencoder
from autoencoder.utils import get_mnist_dataloaders, show_reconstructions, plot_latent_space
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision import transforms
from sklearn.manifold import TSNE
import torch
torch.classes.__path__ = []

# --- Load model ---
@st.cache_resource
def load_model(latent_dim=32, model_path="autoencoder_32.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

# --- Setup ---
st.set_page_config(page_title="Autoencoder Visualizer", layout="centered")
st.title("Autoencoder Visualizer")

# --- Sidebar Controls ---
# latent_dim = st.sidebar.slider("Latent Dimension", min_value=4, max_value=64, step=4, value=32)
# model_path = f"autoencoder_{latent_dim}.pth"
# st.sidebar.write(f"Using model: `{model_path}`")
# --- Sidebar Controls ---
latent_dim = st.sidebar.select_slider(
    "Latent Dimension",
    options=[4, 8, 16, 32, 64],
    value=32
)
model_path = f"models/autoencoder_{latent_dim}.pth"
st.sidebar.write(f"Using model: `{model_path}`")

model, device = load_model(latent_dim=latent_dim, model_path=model_path)

# --- Get Data ---
_, test_loader = get_mnist_dataloaders(batch_size=128)

# --- Main Options ---
tab1, tab2, tab3 = st.tabs(["🔁 Reconstructions", "🧬 Latent Space", "📤 Upload Image"])

# --- Tab 1: Show Reconstructions ---
with tab1:
    st.subheader("Original vs Reconstructed")
    images, _ = next(iter(test_loader))
    images = images.to(device)

    with torch.no_grad():
        recon, _ = model(images)

    num_images = 6
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
    for i in range(num_images):
        axes[0, i].imshow(images[i][0].cpu(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(recon[i][0].cpu(), cmap='gray')
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel("Original", fontsize=14)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=14)
    st.pyplot(fig)

# --- Tab 2: Latent Space ---
with tab2:
    st.subheader("Latent Space Visualization (t-SNE)")

    @st.cache_data
    def compute_tsne_plot(latent_dim):
        from sklearn.manifold import TSNE
        import numpy as np

        model.eval()
        latent_vectors = []
        labels = []

        count = 0
        for images, targets in test_loader:
            images = images.to(device)
            with torch.no_grad():
                _, z = model(images)
            latent_vectors.append(z.cpu().numpy())
            labels.append(targets.numpy())
            count += len(images)
            if count >= 1000:
                break

        latent_vectors = np.concatenate(latent_vectors, axis=0)[:1000]
        labels = np.concatenate(labels, axis=0)[:1000]

        tsne = TSNE(n_components=2, random_state=42)
        z_2d = tsne.fit_transform(latent_vectors)
        return z_2d, labels, latent_vectors, tsne

    with st.spinner("Computing t-SNE..."):
        z_2d, labels, _, _ = compute_tsne_plot(latent_dim)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap='tab10', s=15)
    plt.colorbar(scatter, ax=ax, ticks=range(10))
    ax.set_title("Latent Space (t-SNE)")
    st.pyplot(fig)

# --- Tab 3: Upload Your Own Image ---
with tab3:
    st.subheader("Upload a 28x28 Grayscale Image")

    uploaded_file = st.file_uploader("Choose a PNG image (28x28, grayscale)", type=["png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("L").resize((28, 28))
            st.image(image, caption="Uploaded Image", width=150)

            transform = transforms.ToTensor()
            tensor_img = transform(image).unsqueeze(0).to(device)

            # Encode image to latent space
            with torch.no_grad():
                _, z_custom = model(tensor_img)
            z_custom_np = z_custom.cpu().numpy()

            # t-SNE + projection
            with st.spinner("Computing latent space projection..."):
                z_2d, labels, latent_vectors, tsne_model = compute_tsne_plot(latent_dim)
                z_custom_proj = tsne_model.fit_transform(np.vstack([latent_vectors, z_custom_np]))[-1]

            # Plot with overlay

            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap='tab10', s=15)
            ax.scatter(z_custom_proj[0], z_custom_proj[1], color='red', marker='*', s=200, label='Your Image')
            plt.colorbar(scatter, ax=ax, ticks=range(10))
            ax.set_title("Latent Space with uploaded image")
            st.pyplot(fig)
            # fig, ax = plt.subplots(figsize=(8, 6))
            # scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap='tab10', s=15, alpha=0.6, label='MNIST')
            # ax.legend()
            # ax.set_title("Latent Space with Uploaded Image")
            # st.pyplot(fig)

            # Also show reconstruction
            with torch.no_grad():
                recon, _ = model(tensor_img)

            fig, ax = plt.subplots(1, 2, figsize=(6, 3))
            ax[0].imshow(tensor_img[0][0].cpu(), cmap='gray')
            ax[0].set_title("Original")
            ax[0].axis("off")
            ax[1].imshow(recon[0][0].cpu(), cmap='gray')
            ax[1].set_title("Reconstructed")
            ax[1].axis("off")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error processing image: {e}")
