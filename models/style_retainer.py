import torch

from torchvision import models

class StyleRetainer_VGG(torch.nn.Module):
    def __init__(self):
        super(StyleRetainer_VGG, self).__init__()

        vgg_pretrained_features = models.vgg19(pretrained=True).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

        for x in range(1):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(1,6):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(6,11):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(11, 20):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(20, 29):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):

        relu1_1 = self.slice1(x)
        relu2_1 = self.slice2(relu1_1)
        relu3_1 = self.slice3(relu2_1)
        relu4_1 = self.slice4(relu3_1)
        relu5_1 = self.slice5(relu4_1)

        return {"scale_1": relu1_1,
                "scale_2": relu2_1,
                "scale_3": relu3_1,
                "scale_4": relu4_1,
                "scale_5": relu5_1}

class StyleRetainer_ResNet(torch.nn.Module):
    def __init__(self):
        super(StyleRetainer_ResNet, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

        for x in range(4):
            self.slice1.add_module(str(x), modules[x])
        for x in range(4,5):
            self.slice2.add_module(str(x), modules[x])
        for x in range(5,6):
            self.slice3.add_module(str(x), modules[x])
        for x in range(6,7):
            self.slice4.add_module(str(x), modules[x])
        for x in range(7,8):
            self.slice5.add_module(str(x), modules[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):

        feature1 = self.slice1(x)
        feature2 = self.slice2(feature1)
        feature3 = self.slice3(feature2)
        feature4 = self.slice4(feature3)
        feature5 = self.slice5(feature4)

        return {"scale_1": feature1,
                "scale_2": feature2,
                "scale_3": feature3,
                "scale_4": feature4,
                "scale_5": feature5}


class StyleRetainer_Incep(torch.nn.Module):

    def __init__(self, requires_grad=True):
        super(StyleRetainer_Incep, self).__init__()
        incept = models.inception_v3(pretrained=True)
        modules = list(incept.children())
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()



        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):

        features = self.conv_net(x)

        return features
