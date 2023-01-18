import torch.nn as nn
import torch.nn.functional as F


class MNISTclassifier(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(5, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(5, 5))
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        x = x.view(-1, 512)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
