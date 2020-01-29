import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
from models.classifier import ResnetClassifier2
from models.vgg19 import Vgg19
from models.encoder import ECCVEncoder, Encoder, ResnetEncoder
from models.generator import Generator, ResnetGenerator, ECCVGenerator
from models.discriminator import ECCVDiscriminator, NLayerDiscriminator, PixelDiscriminator
from models.transformer import Transformer
from models.c_attention import Channel_Attention
from models.s_attention import Spatial_Attention
from models.k_attention import Kernel_Attention
from models.k_convolution import K_Convolution
import os
from models.convolution import Convolution, Res_Convolution

###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, lr_policy, epoch_count, niter, niter_decay, lr_decay_iters):
    if lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + epoch_count - niter) / float(niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_iters, gamma=0.1)
    elif lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda(gpu_ids[0])
        # net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net



def define_D(input_nc, ndf, which_model_netD,
             gpu_ids=[]):
    norm = 'batch'
    init_type = 'normal'
    use_sigmoid = False
    norm_layer = get_norm_layer(norm_type=norm)
    n_layers_D = 3

    if which_model_netD == 'patch_GAN':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator(input_nc, ndf, norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'ECCV':
        netD = ECCVDiscriminator(input_nc)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    return init_net(netD, init_type, gpu_ids)


def define_VGG():

    netVGG = Vgg19()
    netVGG.cuda()

    return netVGG


def define_channel_attention(gpu_ids):

    net_CA = Channel_Attention()
    init_type = 'normal'

    return init_net(net_CA, init_type, gpu_ids)


def define_spatial_attention(gpu_ids):

    net_SA = Spatial_Attention()
    init_type = 'normal'

    return init_net(net_SA, init_type, gpu_ids)

def define_kernel_attention(gpu_ids):

    net_KA = Kernel_Attention()
    init_type = 'normal'

    return init_net(net_KA, init_type, gpu_ids)


def define_G(input_nc, output_nc, ngf, gpu_ids):

    norm = 'instance'
    norm_layer = get_norm_layer(norm_type=norm)
    use_dropout = False
    init_type = 'normal'

    netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer, use_dropout=use_dropout, n_blocks=9)

    # save_path = './VGGdecoder/net_G.pth'
    # netG.load_state_dict(torch.load(save_path))

    # netG.cuda()

    return init_net(netG, init_type, gpu_ids)


def define_Convolution(gpu_ids):

    # net_C = Convolution(kernel_size=3)

    net_C = Res_Convolution(kernel_size=3)

    init_type = 'normal'

    return init_net(net_C, init_type, gpu_ids)


def define_K_Convolution(gpu_ids):

    net_KC = K_Convolution(kernel_size=3)

    init_type = 'normal'

    return init_net(net_KC, init_type, gpu_ids)

def define_E(input_nc, ngf, gpu_ids):

    norm = 'instance'
    norm_layer = get_norm_layer(norm_type=norm)
    init_type = 'normal'
    net_E = ResnetEncoder(input_nc, ngf, norm_layer)

    return init_net(net_E, init_type, gpu_ids)