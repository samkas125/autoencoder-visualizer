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
5. **Synthetic Image Generation**: Generate multiple similar images based on an input image by sampling around its latent vector using the `synthetic_generator.py` script.

## Usage

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

## Synthetic Image Generation

The `synthetic_generator.py` script allows you to generate synthetic images based on an input image. It uses the autoencoder as a **variational autoencoder** by sampling points from a normal distribution around the latent vector of the input image.

### How to Use

1. Place your input image in the project directory (e.g., `4.png`).
2. Run the script with the desired parameters:

    ```bash
    python synthetic_generator.py
    ```

3. The generated synthetic images will be saved in the `outputs/synthetic/` directory.

### Parameters - update in `synthetic_generator.py`

- `input_image_path`: Path to the input image.
- `latent_dim`: Dimension of the latent space (default: `32`).
- `model_path`: Path to the pre-trained autoencoder model, based on chosen latent dimension (default: `models/autoencoder_32.pth`).
- `num_images`: Number of synthetic images to generate (default: `10`).

### Example

To generate 10 synthetic images using a latent dimension of 64:

```bash
python synthetic_generator.py
```

## Requirements

- PyTorch
- Streamlit
- scikit-learn
- torchvision
- matplotlib
- Pillow


## Directory Structure

- `autoencoder/`: Contains the model definition, training script, and utility functions.
- `data/`: Stores the MNIST dataset.
- `models/`: Pre-trained autoencoder models for different latent dimensions.
- `outputs/`: Stores visualizations such as t-SNE plots and reconstruction images.
- `streamlit_app.py`: The main Streamlit app for visualization.

## Notes

- Ensure that the `models/` directory contains the pre-trained autoencoder models (e.g., `autoencoder_32.pth`).
- The app dynamically loads the model based on the selected latent dimension in the sidebar.