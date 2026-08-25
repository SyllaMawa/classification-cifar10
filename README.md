# CIFAR-10 Classification from Scratch

A PyTorch implementation of a Convolutional Neural Network (CNN) trained from scratch to classify images from the CIFAR-10 dataset. This project includes data augmentation pipelines, TensorBoard monitoring, and evaluations showing model performance.

## 🎯 Project Overview

### Objective
The goal is to build, train, and evaluate a custom CNN model on the CIFAR-10 dataset without using pre-trained backbones. It highlights deep learning workflow elements: custom dataset preprocessing/augmentation, training loops with validation steps, model checkpoints, tensorboard tracking, and batch inference.

### Key Features
- **Custom CNN Architecture**: A 3-layer convolutional network followed by a fully connected classifier.
- **Data Augmentation**: Incorporates Gaussian blur, horizontal flips, random rotations, random resized crops, and autocontrast transforms to avoid overfitting.
- **TensorBoard Integration**: Logs scalar values (training/validation loss) and images (sample input batches) for visual monitoring of training runs.
- **Save/Load Checkpoints**: Easy saving of model state dicts and verification using inference scripts.

## 📊 Dataset

### CIFAR-10 Dataset
- **Total Images**: 60,000 color images (50,000 for training, 10,000 for testing).
- **Classes**: 10 classes (`plane`, `car`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`).
- **Resolution**: 32×32 pixels, 3 channels (RGB).

## 🏗️ Architecture

The model is defined in `model.py` and consists of:
- **Convolutional Layer 1**: `Conv2d(3 -> 16, kernel_size=3)` + ReLU + MaxPool2d (Output: 16×16×16)
- **Convolutional Layer 2**: `Conv2d(16 -> 32, kernel_size=3)` + ReLU + MaxPool2d (Output: 32×8×8)
- **Convolutional Layer 3**: `Conv2d(32 -> 64, kernel_size=3)` + ReLU + MaxPool2d (Output: 64×4×4)
- **Classifier Layer**: `Linear(64*4*4 -> 10)` mapping to class logits.

## 🚀 Running the Project

### Prerequisites
Install PyTorch, Torchvision, TensorBoard, tqdm, and matplotlib:
```bash
pip install torch torchvision tensorboard tqdm matplotlib
```

### 1. Training the Model
Run `main.py` to start training. The script automatically downloads the CIFAR-10 dataset to local folders, applies data augmentations, runs the training loop for 101 epochs, and logs the stats to TensorBoard:
```bash
python main.py
```

To run TensorBoard:
```bash
tensorboard --logdir=logs/
```

### 2. Testing / Inference
Run `inference.py` to load a saved checkpoint and evaluate model accuracy on the full test set:
```bash
python inference.py
```

### 3. Visualizing Dataset Samples
Run `visu.py` to visualize random augmented batches from the training set:
```bash
python visu.py
```
