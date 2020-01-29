
import torch.nn as nn

class K_Convolution(nn.Module):

    def __init__(self, kernel_size):
        super(K_Convolution, self).__init__()

        reflection_padding = kernel_size // 2

        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv = nn.Conv2d(256, 256, kernel_size=3, stride=1)


    def forward(self, Attention, X):

        self.conv.weight.data = Attention * self.conv.weight.data
        out = self.reflection_pad(X)
        out = self.conv(out)

        return out
