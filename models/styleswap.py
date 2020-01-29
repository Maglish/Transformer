import torch.nn as nn
import torch
from torch.autograd import Variable


class STYLESWAP(nn.Module):

    def __init__(self, styleactivation, shuffle):
        super(STYLESWAP, self).__init__()
        patchsize = 3
        stride = 1
        channel = styleactivation.shape[1]
        self.target_patches = self.extract_patches(styleactivation, patchsize, stride, shuffle)
        npatches = self.target_patches.shape[0]

        self.conv_enc, self.conv_dec, self.aux = self.build(patchsize, stride, channel, npatches)

    def build(self, patchsize, stride, channel, npatches):

        #for each patch, divide by its L2 norm.
        self.enc_patches = self.target_patches.clone()

        for i in range(npatches):
            self.enc_patches[i].mul(1 / (torch.norm(self.enc_patches[i], 2) + 1e-8))

        #Convolution for computing the semi-normalized cross correlation

        conv_enc = nn.Conv2d(channel, npatches, patchsize, stride, bias=False)
        conv_enc.weight.data = self.enc_patches.cpu().data
        for param in conv_enc.parameters():
            param.requires_grad = False
        conv_enc.cuda()

        conv_dec = nn.ConvTranspose2d(npatches, channel, patchsize, stride, bias=False)
        conv_dec.weight.data = self.target_patches.cpu().data
        for param in conv_dec.parameters():
            param.requires_grad = False
        conv_dec.cuda()

        aux = nn.ConvTranspose2d(npatches, channel, patchsize, stride, bias=False)
        aux.weight.data = torch.ones_like(self.target_patches).cpu().data
        for param in aux.parameters():
            param.requires_grad = False
        aux.cuda()

        return conv_enc, conv_dec, aux

    def extract_patches(self, activation, patchsize, stride, shuffle):

        kH, kW = patchsize, patchsize
        dH, dW = stride, stride
        patches = activation.unfold(2, kH, dH).unfold(3, kW, dW).squeeze()
        n1, n2, n3, n4, n5 = patches.shape[0], patches.shape[1], patches.shape[2], patches.shape[3], patches.shape[4]

        patches = patches.permute(1, 2, 0, 3, 4).contiguous().view(n2 * n3, n1, n4, n5)

        if shuffle:

            shuf = Variable(torch.randperm(patches.shape[0]).long()).cuda()
            patches = torch.index_select(patches, 0, shuf)

        return patches

    def forward(self, input):

        similarity = self.conv_enc(input)
        arg_max_filter = torch.max(similarity, 1)
        self.output = torch.zeros_like(similarity)
        self.output = self.output.cuda()
        for i in range(self.output.shape[2]):
            for j in range(self.output.shape[3]):
                ind = arg_max_filter[1][0, i, j]
                self.output[0, ind.cpu().data.numpy()[0], i, j] = 1
        swap = self.conv_dec(self.output)
        swap_wei = self.aux(self.output)

        return swap.div(swap_wei)















