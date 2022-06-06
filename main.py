if __name__ == '__main__':
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
      transform= transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),  # randomly flip and rotate
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

      # loading the images using DataLoader
      train_len = len(dataset) - 100
      test_len = 100
      trainset, testset = torch.utils.data.random_split(dataset, [train_len, test_len])

      trainset = datasets.ImageFolder(data_dir, transform=transform)
      train_loader = torch.utils.data.DataLoader(trainset, batch_size=40, shuffle=True, num_workers=2)

      testset = datasets.ImageFolder(data_dir, transform=transform)
      test_loader = torch.utils.data.DataLoader(testset, batch_size=40, shuffle=False, num_workers=2)

      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

      class CNN(nn.Module):
            def __init__(self):
                  super(CNN, self).__init__()
                  self.conv_layer = nn.Sequential(
                        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
                        nn.BatchNorm2d(32),
                        nn.LeakyReLU(inplace=True),
                        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
                        nn.BatchNorm2d(32),
                        nn.LeakyReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.LeakyReLU(inplace=True),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.LeakyReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                  )
                  self.fc_layer = nn.Sequential(
                        nn.Dropout(p=0.1),
                        nn.Linear(8 * 8 * 64, 1000),
                        nn.ReLU(inplace=True),
                        nn.Linear(1000, 512),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=0.1),
                        nn.Linear(512, 10)
                  )

            def forward(self, x):
                  # conv layers
                  x = self.conv_layer(x)
                  # flatten
                  x = x.view(x.size(0), -1)
                  # fc layer
                  x = self.fc_layer(x)
                  return x


      model = CNN()

      criterion = nn.CrossEntropyLoss()
      optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

      total_step = len(trainset)
      loss_list = []
      acc_list = []

      for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(trainset):
                  # Forward pass
                  outputs = model(images)
                  loss = criterion(outputs, labels)
                  loss_list.append(loss.item())
                  # Backprop and optimisation
                  optimizer.zero_grad()
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

      model.eval()

      # with torch.no_grad():
      #       correct = 0
      #       total = 0
      #       for (images, labels) in test_loader:
      #             outputs = model(images)
      #             _, predicted = torch.max(outputs.data, 1)
      #             total += labels.size(0)
      #             correct += (predicted == labels).sum().item()
      #       print('Test Accuracy of the model on the 10000 test images: {} %'
      #             .format((correct / total) * 100))
