import os

from PIL import Image

if __name__ == '__main__':
      import torch
      import torchvision.transforms as transforms
      import torchvision.datasets as datasets
      import torch.nn as nn
      import torch.nn.functional as F
      import torch.utils.data as td
      from torchvision.datasets import ImageFolder
      from torchvision.transforms import ToTensor
      import sys
      import warnings

      if not sys.warnoptions:
            warnings.simplefilter("ignore")

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
      path = './dataset'
      for folder in os.listdir(path):
            sub_folder_path = os.path.join(path,folder)
            for mask_folder in os.listdir(sub_folder_path):
                  mask_sets = os.path.join(sub_folder_path,mask_folder)
                  for file in  os.listdir(mask_sets):
                        file_path = os.path.join(mask_sets,file)
                        im=Image.open(file_path)
                        rbg_im = im.convert('RGB')
      print("Image conversion done\n")
      dataset = ImageFolder('./dataset', transform=ToTensor())
      print("- The dataset has classes", dataset.classes,
            "and contains", len(dataset), "images")

      num_epochs = 4
      num_classes = 4
      learning_rate = 0.001

      # transform process the images (resizing and normalizing)
      transform= transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.1, 0.1, 0.1), (0.1, 0.1, 0.1))])

      training_set = datasets.ImageFolder(root='./dataset/Training', transform=transform)
      train_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True, num_workers=2)
      print("- The training dataset has classes", training_set.classes,
            "and contains", len(training_set), "images")

      testing_set = datasets.ImageFolder(root='./dataset/Testing', transform=transform)
      test_loader = torch.utils.data.DataLoader(testing_set, batch_size=4, shuffle=False, num_workers=2)
      print("- The testing dataset has classes", testing_set.classes,
            "and contains", len(testing_set), "images")

      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      print("The device used is", device)

      classes = ('No Mask', 'Cloth Mask', 'N95 Mask', 'Surgical Mask')

      class CNN(nn.Module):
            def __init__(self):
                  super(CNN, self).__init__()
                  self.conv1 = nn.Conv2d(3, 6, 5)
                  self.pool = nn.MaxPool2d(2, 2)
                  self.conv2 = nn.Conv2d(6, 16, 5)
                  self.fc1 = nn.Linear(16 * 13 * 13, 120)
                  self.fc2 = nn.Linear(120, 84)
                  self.fc3 = nn.Linear(84, 10)

            def forward(self, x):
                  x = self.pool(F.relu(self.conv1(x)))
                  x = self.pool(F.relu(self.conv2(x)))
                  x = x.view(4, 16 * 13 * 13)
                  x = F.relu(self.fc1(x))
                  x = F.relu(self.fc2(x))
                  x = self.fc3(x)
                  return x

      model = CNN().to(device)

      criterion = nn.CrossEntropyLoss()
      optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

      total_step = len(train_loader)
      loss_list = []
      acc_list = []

      for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader, 0):

                  images, labels = images.to(device), labels.to(device)

                  # Forward pass
                  outputs = model(images)
                  loss = criterion(outputs, labels)
                  loss_list.append(loss.item())

                  # Backprop and optimisation
                  loss.backward()
                  optimizer.step()

                  # Train accuracy
                  total = labels.size(0)
                  _, predicted = torch.max(outputs.data, 1)
                  correct = (predicted == labels).sum().item()
                  acc_list.append(correct / total)
                  if (i + 1) % 100 == 0:
                        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                              .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), (correct / total) * 100))

      print('Training Done')

      with torch.no_grad():
            correct = 0
            total = 0
            for (images, labels) in test_loader:
                  outputs = model(images)
                  _, predicted = torch.max(outputs.data, 1)
                  total += labels.size(0)
                  correct += (predicted == labels).sum().item()
            print('Test Accuracy of the model on the 400 test images: {} %'
                  .format((correct / total) * 100))
