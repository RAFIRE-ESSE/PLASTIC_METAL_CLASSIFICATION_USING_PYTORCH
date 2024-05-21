import cv2
import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.ImageFolder(root='./train_devil', transform=transform)
trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

classes = trainset.classes
print(classes)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(35344,4)
        self.fc2 = nn.Linear(4, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
net.load_state_dict(torch.load("./devil_net.pth"))
z=cv2.resize(cv2.imread("archives-ewaste-one_one.jpg"),(200,200))
z=TF.to_tensor(z)
z.unsqueeze_(0)
_, predicted = (torch.max(net(z), 1))
print(predicted)
print('Predicted: ', ' '.join(f'{classes[predicted[0]]:5s}'))

