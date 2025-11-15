# src/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    Simple CNN with dynamic computation of the flattened feature size.
    Use: SimpleCNN(num_classes=10, in_channels=3, input_size=(32,32))
    """
    def __init__(self, num_classes=10, in_channels=3, input_size=(32, 32)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 5, padding=2)
        self.pool1 = nn.MaxPool2d(3, stride=2)
        self.conv2 = nn.Conv2d(64, 64, 5, padding=2)
        self.pool2 = nn.MaxPool2d(3, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 5, padding=2)
        self.pool3 = nn.MaxPool2d(3, stride=2)

        self._feature_dim = self._get_feature_dim(input_size, in_channels)

        self.fc1 = nn.Linear(self._feature_dim, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, num_classes)

    def _get_feature_dim(self, input_size, in_channels):
        h, w = input_size
        with torch.no_grad():
            x = torch.zeros(1, in_channels, h, w)
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
            x = F.relu(self.conv3(x))
            x = self.pool3(x)
            feat_dim = x.view(1, -1).shape[1]
        return feat_dim

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
