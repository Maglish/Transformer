import torch.nn as nn
import torchvision
import torch

class ResnetClassifier2(nn.Module): #default initialize the last layer
    def __init__(self):
        super(ResnetClassifier2, self).__init__()
        self.basenet = torchvision.models.resnet50(pretrained=False, num_classes = 4)
        # self.basenet.fc = nn.Sequential(nn.Linear(2048, 27))

    def forward(self, x):


        return self.basenet(x)