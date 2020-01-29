
import torch.nn as nn

class Spatial_Attention(nn.Module):

    def __init__(self):
        super(Spatial_Attention, self).__init__()

        self.conv1 = nn.Conv2d(256, 256, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(256, 256, kernel_size=1, stride=1)
        self.sig = nn.Sigmoid()


    def forward(self, X):

        X = self.conv1(X)
        X = self.relu(X)
        X = self.conv2(X)
        X = self.sig(X)

        return X
