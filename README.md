# Autoencoder Visualizer

This project provides an interactive visualization tool for exploring the latent space of autoencoder models trained on the MNIST dataset. The app allows users to:

- Compare original and reconstructed images.
- Visualize the latent space using t-SNE.
- Upload custom images to see their reconstructions and their position in the latent space.

## Features

1. **Reconstructions**: View side-by-side comparisons of original and reconstructed images from the MNIST test set.
2. **Latent Space Visualization**: Explore the 2D t-SNE projection of the latent space, color-coded by digit labels.
3. **Custom Image Upload**: Upload a 28x28 grayscale image to see its reconstruction and its position in the latent space.
4. **Pre-trained models** for MNIST dataset with varying dimensions for latent space. Observe differences in quality of reconstructed output based on dimension of latent space!

## Requirements

- PyTorch
- Streamlit
- scikit-learn
- torchvision
- matplotlib
- Pillow

1. Clone the repository and navigate to the project directory:

```bash
git clone https://www.github.com/samkas125/autoencoder-visualizer
cd autoencoder_visualizer
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

4. Open the app in your browser. By default, it will be available at `http://localhost:8501`.

## Directory Structure

- `autoencoder/`: Contains the model definition, training script, and utility functions.
- `data/`: Stores the MNIST dataset.
- `models/`: Pre-trained autoencoder models for different latent dimensions.
- `outputs/`: Stores visualizations such as t-SNE plots and reconstruction images.
- `streamlit_app.py`: The main Streamlit app for visualization.

## Notes

- Ensure that the `models/` directory contains the pre-trained autoencoder models (e.g., `autoencoder_32.pth`).
- The app dynamically loads the model based on the selected latent dimension in the sidebar.