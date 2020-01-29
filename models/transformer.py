import torch

class Transformer(torch.nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=9, stride=1, padding=4)
        self.netT = torch.nn.utils.weight_norm(self.conv1)

    def forward(self, input):

        return self.netT(input)
