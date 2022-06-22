import warnings
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import KFold

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
      import seaborn as sns 


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
      k=10
      num_classes = 2
      learning_rate = 0.000001

      # transform process the images (resizing and normalizing)
      transform= transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

      age_set = datasets.ImageFolder(root='./dataset/Age Training', transform=transform)

      testing_size = len(age_set) * 0.2
      training_size = len(age_set) - (testing_size)

      training_set, testing_set = torch.utils.data.random_split(
            age_set, [int(training_size), int(testing_size)]
      )

      print("- The age-based training dataset has classes", age_set.classes, "and contains", len(age_set),
            "images")

      train_loader = torch.utils.data.DataLoader(training_set, batch_size=2, shuffle=True, num_workers=2)
      print("- The training dataset contains", len(training_set), "images")
      test_loader = torch.utils.data.DataLoader(testing_set, batch_size=2, shuffle=False, num_workers=2)
      print("- The testing dataset contains", len(testing_set), "images")


      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      print("The device used is", device)

      print()
      print('                  Training Part                      ')
      print('-----------------------------------------------------')

      m = len(age_set)
      train_data, val_data = random_split(age_set, [int(m - m * 0.2), int(m * 0.2)])

      y_train = np.array([y for x, y in iter(train_data)])

      classes = ('Old', 'Young')

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
                        nn.Linear(512, 2)
                  )

            def forward(self, x):
                  x = self.conv_layer(x)
                  x = x.view(x.size(0), -1)
                  x = self.fc_layer(x)
                  return x

      model = CNN().to(device)

      criterion = nn.CrossEntropyLoss()
      optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

      # total_step = len(train_loader)
      # loss_list = []
      # acc_list = []


      # for epoch in range(num_epochs):
      #       for i, (images, labels) in enumerate(train_loader, 0):

      #             images, labels = images.to(device), labels.to(device)

      #             # Forward pass
      #             outputs = model(images)
      #             loss = criterion(outputs, labels)
      #             loss_list.append(loss.item())

      #             # Backprop and optimisation
      #             loss.backward()
      #             optimizer.step()

      #             # Train accuracy
      #             total = labels.size(0)
      #             _, predicted = torch.max(outputs.data, 1)
      #             correct = (predicted == labels).sum().item()
      #             acc_list.append(correct / total)
      #             if (i + 1) % 100 == 0:
      #                   print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%, Class:[{}]'
      #                         .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), (correct / total) * 100, age_set.classes[epoch]))
      # print('Training Done')



      # model.eval()
      # with torch.no_grad():
      #       correct = 0
      #       total = 0
      #       for (images, labels) in test_loader:
      #             outputs = model(images)
      #             _, predicted = torch.max(outputs.data, 1)
      #             total += labels.size(0)
      #             correct += (predicted == labels).sum().item()
      #       print('Test Accuracy of the model on the 400 test images: {} %'
      #             .format((correct / total) * 100))
  
      # torch.save(model.state_dict(), "Trained_Age")
      splits=KFold(n_splits=k,shuffle=True,random_state=42)
      foldperf={}

      def train_epoch(model,device,dataloader,loss_fn,optimizer):
        train_loss,train_correct=0.0,0
        model.train()
        for images, labels in dataloader:

            images,labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output,labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            scores, predictions = torch.max(output.data, 1)
            train_correct += (predictions == labels).sum().item()

        return train_loss,train_correct
  
      def valid_epoch(model,device,dataloader,loss_fn):
        valid_loss, val_correct = 0.0, 0
        model.eval()
        for images, labels in dataloader:

            images,labels = images.to(device),labels.to(device)
            output = model(images)
            loss=loss_fn(output,labels)
            valid_loss+=loss.item()*images.size(0)
            scores, predictions = torch.max(output.data,1)
            val_correct+=(predictions == labels).sum().item()
          

        return valid_loss,val_correct
    
      def get_metrics(model, dataloader):
         model.eval()
         prediction_list = []
         accurate_list = []
         with torch.no_grad():
          for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(model(images), 1)
            prediction_list.extend(predicted.detach().cpu().numpy())
            accurate_list.extend(labels.detach().cpu().numpy())
          print()
          print("Classification Report: ")
          print(classification_report(prediction_list, accurate_list, target_names = ['Old', 'Young']))

          confusion_matrix_data = confusion_matrix(accurate_list, prediction_list)
          conf_matrix = sns.heatmap(confusion_matrix_data, annot=True, fmt='g' )

          conf_matrix.set_title('Confusion Matrix');
          conf_matrix.set_xlabel('Predicted Categories')
          conf_matrix.set_ylabel('Actual Categories');

          conf_matrix.xaxis.set_ticklabels(["Old", "Young"])
          conf_matrix.yaxis.set_ticklabels(["Old", "Young"])

      for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(age_set)))):

        print()
        print('Fold {}'.format(fold + 1))

        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        test_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
        train_loader = torch.utils.data.DataLoader(age_set, batch_size=32, sampler=train_sampler)
        test_loader = torch.utils.data.DataLoader(age_set, batch_size=32, sampler=test_sampler)
    
    

        history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}

        for epoch in range(num_epochs):
            train_loss, train_correct=train_epoch(model,device,train_loader,criterion,optimizer)
            test_loss, test_correct=valid_epoch(model,device,test_loader,criterion)

            train_loss = train_loss / len(train_loader.sampler)
            train_acc = train_correct / len(train_loader.sampler) * 100
            test_loss = test_loss / len(test_loader.sampler)
            test_acc = test_correct / len(test_loader.sampler) * 100

            print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(epoch + 1,
                                                                                                                num_epochs,
                                                                                                                train_loss,
                                                                                                                test_loss,
                                                                                                                train_acc,
                                                                                                                test_acc))
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            foldperf['fold{}'.format(fold+1)] = history  

      torch.save(model,'k_cross_Age.pt')   


      testl_f,tl_f,testa_f,ta_f=[],[],[],[]
      k=10
      for f in range(1,k+1):

        tl_f.append(np.mean(foldperf['fold{}'.format(f)]['train_loss']))
        testl_f.append(np.mean(foldperf['fold{}'.format(f)]['test_loss']))

        ta_f.append(np.mean(foldperf['fold{}'.format(f)]['train_acc']))
        testa_f.append(np.mean(foldperf['fold{}'.format(f)]['test_acc']))

      print('Performance of {} fold cross validation'.format(k))
      print("Average Training Loss: {:.3f} \t Average Test Loss: {:.3f} \t Average Training Acc: {:.2f} \t Average Test Acc: {:.2f}".format(np.mean(tl_f),np.mean(testl_f),np.mean(ta_f),np.mean(testa_f)))     
      print()

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
      batch_size=4,
      optimizer=optim.Adam,
      criterion=nn.CrossEntropyLoss,
      device= torch.device("cpu")
      )


      # net.fit(train_data, y = y_train)
      # y_pred = net.predict(testing_set)
      # y_test = np.array([y for x, y in iter(testing_set)])

      # acc_score = accuracy_score(y_test, y_pred)
      # f1 = f1_score(y_test, y_pred, average="macro")
      # recall = recall_score(y_test, y_pred, average="macro")
      # precision = precision_score(y_test, y_pred, average="macro")

      # print(f"The accuracy score of the test set: {acc_score: .2f}")
      # print(f"The f1-score of the test set is: {f1: .2f}")
      # print(f"The recall of the test set is: {recall: .2f}")
      # print(f"The precision of the test set is: {precision: .2f}")
      # plot_confusion_matrix(net, testing_set, y_test.reshape(-1, 1), display_labels = ['Old', 'Young'])

      get_metrics(model,test_loader)
      plt.show()