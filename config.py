import os
import torch
from torchvision import transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
data_path = '/kaggle/input/weather-dataset/dataset'
output_dir = '/kaggle/working'

# Image settings
IMG_SIZE = 224
batch_size = 8
num_worker = os.cpu_count()

# Training hyperparameters
num_epochs = 20
lr = 0.0001
patience = 5
step_size = 1
gamma = 0.1

# Data transformations
transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

transform_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Train-test split ratios
ratios = [0.8, 0.7, 0.6]