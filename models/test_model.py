from torch.autograd import Variable
from . import networks
import os
import torch
from collections import OrderedDict

class TestModel():

    def __init__(self, name, which_epoch):


        self.gpu_ids = [0]
        self.isTrain = True
        self.continue_train = False
        self.checkpoints_dir = './checkpoints'
        self.which_epoch = which_epoch  # which epoch to load? set to latest to use latest cached model
        self.name = name
        self.save_dir = os.path.join(self.checkpoints_dir, self.name)
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')

        self.model_names = ['G', 'E']
        self.visual_names = ['real_A', 'R']
        self.input_nc = 3
        self.output_nc = 3
        self.ndf = 64  # number of filters in the first layer of discriminator
        self.ngf = 32

        self.which_model_netG = 'resnet_9blocks'
        self.which_model_netE = 'resnet_blocks'

        self.netG = networks.define_G(self.input_nc, self.output_nc, self.ngf,
                                      self.which_model_netG, self.gpu_ids)

        self.netE = networks.define_E(self.input_nc, self.ngf,
                                      self.which_model_netE, self.gpu_ids)


        self.load_networks(self.which_epoch)
        self.print_networks()

    def set_input(self, input):
        # we need to use single_dataset mode
        input_A = input['A']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
        self.input_A = input_A
        self.image_paths = input['A_paths']

    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.latent = self.netE(self.real_A)
        self.R = self.netG(self.latent)


    def load_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location=self.device)

                net.load_state_dict(state_dict)

    def print_networks(self):

        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_image_paths(self):

        return self.image_paths