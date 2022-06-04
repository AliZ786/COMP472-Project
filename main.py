import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import torchvision.datasets

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
print("- The dataset has classes", dataset.classes,
      "and contains", len(dataset), "images")

num_epochs = 4
num_classes = 4
learning_rate = 0.001

# transform process the images (resizing and normalizing)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# loading the images using DataLoader
trainset = torchvision.datasets.ImageFolder(root='./dataset', transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=300, shuffle=True, num_workers=2)

testset = torchvision.datasets.ImageFolder(root='./dataset', transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# defining classes
classes = ('No mask', 'N95', 'Surgical', 'Cloth')

