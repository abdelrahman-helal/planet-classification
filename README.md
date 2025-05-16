# Planet Classification

A deep learning project that implements and compares two neural network models for classifying images of solar system planets.

## About the Project

This project implements two different neural network architectures to classify planet images:
1. A Dense Neural Network with three layers
2. A Convolutional Neural Network (CNN) with two convolutional layers and two dense layers

## Dataset

The project uses a public dataset collected and curated by Emirhan BULUT, available at [Planets and Moons Dataset](https://github.com/emirhanai/Planets-and-Moons-Dataset-AI-in-Space). The dataset contains:
- 1192 images across 8 planet classes
- Image dimensions: 144x256x3 (Height x Width x Channels)
- All images are in supported formats (JPEG, PNG, BMP, GIF)

## Model Architectures

### Dense Neural Network
- Input layer: Flattened image data with rescaling
- Hidden layer: 128 neurons with ReLU activation
- Dropout layer (0.2) for regularization
- Output layer: 8 neurons (one for each planet) with linear activation

### Convolutional Neural Network
- Two Conv2D layers (64 and 128 filters) with ReLU activation
- Batch normalization after each Conv2D layer
- MaxPooling2D layers for dimensionality reduction
- Dense layer with 64 neurons and ReLU activation
- Dropout layer (0.2) for regularization
- Output layer: 8 neurons with linear activation

## Results

Both models achieved excellent performance:
- Both reached 100% validation accuracy
- The CNN model converged faster and showed more stability after convergence
- Early stopping was implemented to prevent overfitting

## Dependencies

- TensorFlow
- OpenCV (cv2)
- NumPy
- Pandas
- Matplotlib
- TensorFlow Datasets

## Usage

The project is implemented in a Jupyter notebook (Planet_Classification.ipynb) and can be run in Google Colab or any Jupyter environment with the required dependencies installed.