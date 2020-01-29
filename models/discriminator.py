import torch
import torch.nn as nn
import functools
# ECCV discriminator

class ECCVDiscriminator(torch.nn.Module):
    def __init__(self, input_nc, ngf=64):
        super(ECCVDiscriminator, self).__init__()

        self.conv1 = torch.nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1)
        self.in1 = torch.nn.InstanceNorm2d(ngf, affine=True)
        self.lrelu = torch.nn.LeakyReLU(0.2, True)

        self.aconv1 = torch.nn.Conv2d(ngf, 1, kernel_size=5, stride=1, padding=1)

        self.conv2 = torch.nn.Conv2d(ngf, ngf*2, kernel_size=4, stride=2, padding=1)
        self.in2 = torch.nn.InstanceNorm2d(ngf*2, affine=True)

        self.aconv2 = torch.nn.Conv2d(ngf*2, 1, kernel_size=9, stride=1, padding=4)

        self.conv3 = torch.nn.Conv2d(ngf*2, ngf*4, kernel_size=4, stride=2, padding=1)
        self.in3 = torch.nn.InstanceNorm2d(ngf*4, affine=True)

        self.aconv3 = torch.nn.Conv2d(ngf*4, 1, kernel_size=9, stride=1, padding=4)

        self.conv4 = torch.nn.Conv2d(ngf*4, ngf*8, kernel_size=4, stride=2, padding=1)
        self.in4 = torch.nn.InstanceNorm2d(ngf*8, affine=True)

        self.aconv4 = torch.nn.Conv2d(ngf*8, 1, kernel_size=5, stride=1, padding=2)

        self.conv5 = torch.nn.Conv2d(ngf * 8, ngf * 16, kernel_size=4, stride=2, padding=1)
        self.in5 = torch.nn.InstanceNorm2d(ngf * 16, affine=True)

        self.aconv5 = torch.nn.Conv2d(ngf*16, 1, kernel_size=3, stride=1, padding=1)
        self.sig = torch.nn.Sigmoid()


    def forward(self, input):

        h1 = self.lrelu(self.in1(self.conv1(input)))
        h1_pred = self.sig(self.aconv1(h1))

        h2 = self.lrelu(self.in2(self.conv2(h1)))
        h2_pred = self.sig(self.aconv2(h2))

        h3 = self.lrelu(self.in3(self.conv3(h2)))
        h3_pred = self.sig(self.aconv3(h3))

        h4 = self.lrelu(self.in4(self.conv4(h3)))
        h4_pred = self.sig(self.aconv4(h4))

        h5 = self.lrelu(self.in5(self.conv5(h4)))
        h5_pred = self.sig(self.aconv5(h5))

        return {"scale_1": h1_pred,
                "scale_2": h2_pred,
                "scale_3": h3_pred,
                "scale_4": h4_pred,
                "scale_5": h5_pred
                }

class Discriminator(torch.nn.Module):
    def __init__(self, input_nc, ngf=64):
        super(Discriminator, self).__init__()

        self.conv1 = torch.nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1)
        self.in1 = torch.nn.InstanceNorm2d(ngf, affine=True)
        self.lrelu = torch.nn.LeakyReLU(0.2, True)
        self.aconv1 = torch.nn.Conv2d(ngf, 1, kernel_size=1, stride=1, padding=0)

        self.conv2 = torch.nn.Conv2d(ngf, ngf*2, kernel_size=4, stride=2, padding=1)
        self.in2 = torch.nn.InstanceNorm2d(ngf*2, affine=True)
        self.aconv2 = torch.nn.Conv2d(ngf*2, 1, kernel_size=1, stride=1, padding=0)

        self.conv3 = torch.nn.Conv2d(ngf*2, ngf*4, kernel_size=4, stride=2, padding=1)
        self.in3 = torch.nn.InstanceNorm2d(ngf*4, affine=True)
        self.aconv3 = torch.nn.Conv2d(ngf*4, 1, kernel_size=1, stride=1, padding=0)

        self.conv4 = torch.nn.Conv2d(ngf*4, ngf*8, kernel_size=4, stride=2, padding=1)
        self.in4 = torch.nn.InstanceNorm2d(ngf*8, affine=True)
        self.aconv4 = torch.nn.Conv2d(ngf*8, 1, kernel_size=1, stride=1, padding=0)

        self.conv5 = torch.nn.Conv2d(ngf * 8, ngf * 16, kernel_size=4, stride=2, padding=1)
        self.in5 = torch.nn.InstanceNorm2d(ngf * 16, affine=True)
        self.aconv5 = torch.nn.Conv2d(ngf*16, 1, kernel_size=1, stride=1, padding=0)

        self.sig = torch.nn.Sigmoid()


    def forward(self, input):

        h1 = self.lrelu(self.in1(self.conv1(input)))
        h1_pred = self.sig(self.aconv1(h1))

        h2 = self.lrelu(self.in2(self.conv2(h1)))
        h2_pred = self.sig(self.aconv2(h2))

        h3 = self.lrelu(self.in3(self.conv3(h2)))
        h3_pred = self.sig(self.aconv3(h3))

        h4 = self.lrelu(self.in4(self.conv4(h3)))
        h4_pred = self.sig(self.aconv4(h4))

        h5 = self.lrelu(self.in5(self.conv5(h4)))
        h5_pred = self.sig(self.aconv5(h5))

        return {"scale_1": h1_pred,
                "scale_2": h2_pred,
                "scale_3": h3_pred,
                "scale_4": h4_pred,
                "scale_5": h5_pred
                }


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)