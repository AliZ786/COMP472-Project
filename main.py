import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

# Install PyTorch first

# To download torch, create the following environment using Anaconda:
'conda create -n pytorch python=3.6'

# Then activate the environment by running
'conda activate pytorch'

# Install using pip torch & torch vision
'pip install torchvision --user'

# Open python shell and import the following
'import torch'
'import torchvision'

# Then run this app using anaconda
'python main.py'

print('-----------------------------------------------------')
print('               Face Mask Detection App               ')
print('-----------------------------------------------------')
print('')
print('Images Collected Statistics:')

# dataset
data_dir = './dataset'
dataset = ImageFolder(data_dir, transform=ToTensor())
print("- The dataset has 2 classes", dataset.classes,
      "and contains", len(dataset), "images")

# Training data
training_dir = './dataset/Training'
training_dataset = ImageFolder(training_dir, transform=ToTensor())
print("- The training dataset has the classes", training_dataset.classes,
      "and contains", len(training_dataset), "images")

# Testing data
testing_dir = './dataset/Testing'
testing_dataset = ImageFolder(testing_dir, transform=ToTensor())
print("- The testing dataset has the classes", testing_dataset.classes,
      "and contains", len(testing_dataset), "images")

# Processing the images (resizing and normalizing)
dataset = ImageFolder(data_dir, transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(*([0.485, 0.456, 0.406], [0.225, 0.225, 0.225]))
]))

# Loading the images using DataLoader
train_loader = torch.utils.data.DataLoader(
    training_dir, batch_size=300, shuffle=True, pin_memory=True)
test_loader = torch.utils.data.DataLoader(
    testing_dir, batch_size=100, pin_memory=True)
