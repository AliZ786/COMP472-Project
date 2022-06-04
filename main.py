from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

# To download torch, create the following environment:
# conda create -n pytorch python=3.6

# Then activate the environment by running
# conda activate pytorch

# Then run this app
print('-----------------------------------------------------')
print('               Face Mask Detection App               ')
print('-----------------------------------------------------')

data_dir = './dataset'
dataset = ImageFolder(data_dir, transform=ToTensor())
print("The dataset has 2 classes", dataset.classes,
      "and contains", len(dataset), "images")

testing_dir = './dataset/Testing'
testing_dataset = ImageFolder(testing_dir, transform=ToTensor())
print("Class", testing_dataset.classes,
      "inside Testing Folder contains", len(testing_dataset), "images")

training_dir = './dataset/Training'
training_dataset = ImageFolder(training_dir, transform=ToTensor())
print("Class", training_dataset.classes,
      "inside Training Folder contains", len(training_dataset), "images")
