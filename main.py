import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
      import os
      import torch
      import torchvision.transforms as transforms
      import torchvision.datasets as datasets
      import torch.nn as nn
      import torch.utils.data as td
      from torchvision.datasets import ImageFolder
      from torchvision.transforms import ToTensor
      from sklearn.metrics import accuracy_score
      from sklearn.metrics import plot_confusion_matrix
      from skorch import NeuralNetClassifier
      import torch.optim as optim
      import numpy as np
      import matplotlib.pyplot as plt
      from sklearn.metrics import f1_score
      from sklearn.metrics import recall_score
      from sklearn.metrics import precision_score
      from torch.utils.data import random_split


      print('-----------------------------------------------------')
      print('               Face Mask Detection App               ')
      print('-----------------------------------------------------')
      print('')

      print('           Images Collected Statistics               ')
      print('-----------------------------------------------------')

      dataset = ImageFolder('./dataset', transform=ToTensor())
      print("- The dataset has classes", dataset.classes,
            "and contains", len(dataset), "images")

      num_epochs = 4
      num_classes = 4
      learning_rate = 0.000001

      # transform process the images (resizing and normalizing)
      transform= transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

      training_set = datasets.ImageFolder(root='./dataset/Training', transform=transform)
      train_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True, num_workers=2)
      print("- The training dataset has classes", training_set.classes, "and contains", len(training_set), "images")

      train_age_set = datasets.ImageFolder(root='./dataset/Age Training', transform=transform)
      train_age_loader = torch.utils.data.DataLoader(train_age_set, batch_size=4, shuffle=True, num_workers=2)
      print("- The age-based training dataset has classes", train_age_set.classes, "and contains", len(train_age_set), "images")

      train_gender_set = datasets.ImageFolder(root='./dataset/Gender Training', transform=transform)
      train_gender_loader = torch.utils.data.DataLoader(train_gender_set, batch_size=4, shuffle=True, num_workers=2)
      print("- The gender-based training dataset has classes", train_gender_set.classes, "and contains", len(train_gender_set),
            "images")

      testing_set = datasets.ImageFolder(root='./dataset/Testing', transform=transform)
      test_loader = torch.utils.data.DataLoader(testing_set, batch_size=4, shuffle=False, num_workers=2)
      print("- The testing dataset has classes", testing_set.classes, "and contains", len(testing_set), "images")

      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      print("The device used is", device)

      print()
      print('                  Training Part                      ')
      print('-----------------------------------------------------')

      m = len(training_set)
      train_data, val_data = random_split(training_set, [int(m - m * 0.2), int(m * 0.2)])

      y_train = np.array([y for x, y in iter(train_data)])

      classes = ('No Mask', 'Cloth Mask', 'N95 Mask', 'Surgical Mask')

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
                        nn.Linear(32 * 32 * 4, 1000),
                        nn.ReLU(inplace=True),
                        nn.Linear(1000, 512),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=0.1),
                        nn.Linear(512, 4)
                  )

            def forward(self, x):
                  x = self.conv_layer(x)
                  x = x.view(x.size(0), -1)
                  x = self.fc_layer(x)
                  return x

      model = CNN().to(device)

      criterion = nn.CrossEntropyLoss()
      optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
                        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%, Class:[{}]'
                              .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), (correct / total) * 100, training_set.classes[epoch]))
      print('Training Done')



      model.eval()
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

      torch.save(model.state_dict(), "Trained_Model")


      print()
      print('                   Evaluation Part                   ')
      print('-----------------------------------------------------')

      torch.manual_seed(0)
      net = NeuralNetClassifier(
      CNN,
      max_epochs=4,
      iterator_train__num_workers=4,
      iterator_valid__num_workers=4,
      lr=0.001,
      batch_size=64,
      optimizer=optim.Adam,
      criterion=nn.CrossEntropyLoss,
      device= torch.device("cpu")
      )

      cloth_train = "./dataset/Training/Cloth Mask"
      n95_train = "./dataset/Training/N95 Mask"
      no_train = "./dataset/Training/No Mask"
      surgical_train = "./dataset/Training/Surgical Mask"

      
      totalClothFiles = 0
      totalN95Files = 0
      totalNoFiles = 0
      totalSurgicalFiles = 0

      for base, dirs, files in os.walk(cloth_train):
       for Files in files:
        totalClothFiles += 1

      for base, dirs, files in os.walk(n95_train):
       for Files in files:
        totalN95Files += 1

      for base, dirs, files in os.walk(no_train):
       for Files in files:
        totalNoFiles += 1

      for base, dirs, files in os.walk(surgical_train):
       for Files in files:
        totalSurgicalFiles += 1

      x = range(4)

      x_labels = ["Cloth", "N95", "No", "Surgical"]

      y = [totalClothFiles, totalN95Files, totalNoFiles, totalSurgicalFiles]

      plt.bar(x,y, color=['black', 'red', 'blue', 'green'])

      plt.title("Number of masks per type")
      plt.ylabel("Total number of masks per type", fontsize =10)
      plt.xticks(x, x_labels)


      for index, data in enumerate(y):
        plt.text(x=index, y=data+1, s=f"{data}",
                  fontdict=dict(fontsize=12, color='maroon'))

      # plt.savefig("MaskDistribution_Train.pdf")




      net.fit(train_data, y = y_train)
      y_pred = net.predict(testing_set)
      y_test = np.array([y for x, y in iter(testing_set)])

      acc_score = accuracy_score(y_test, y_pred)
      f1 = f1_score(y_test, y_pred, average="macro")
      recall = recall_score(y_test, y_pred, average="macro")
      precision = precision_score(y_test, y_pred, average="macro")

      print(f"The accuracy score of the test set: {acc_score: .2f}")
      print(f"The f1-score of the test set is: {f1: .2f}")
      print(f"The recall of the test set is: {recall: .2f}")
      print(f"The precision of the test set is: {precision: .2f}")
      plot_confusion_matrix(net, testing_set, y_test.reshape(-1, 1), display_labels = ['No Mask', 'N95 Mask', 'Cloth Mask', 'Surgical Mask'])
      plt.show()

