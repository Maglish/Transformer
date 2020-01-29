import torch
from torch.autograd import Variable
from models import networks
import os
from collections import OrderedDict
from util.util import gram_matrix
from util.util import normalize_batch, calc_mean_std
from models.loss_function import cal_mean_std
from models.convolution import Convolution

class StyleModel:

    def __init__(self, args):

        self.gpu_ids=[0]
        self.isTrain = True
        
        self.checkpoints_dir = './checkpoints'
        self.which_epoch = 'latest' # which epoch to load? set to latest to use latest cached model
        self.args = args
        # self.name = 'G_GAN_%s_lambdar_%s_lambdas_%s_alpha_%s' % (self.args.lambda_d, self.args.lambda_r, self.args.lambda_s, self.args.alpha)
        self.name = 'Res_convolution_Gram'
        expr_dir = os.path.join(self.checkpoints_dir, self.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)

        self.save_dir = os.path.join(self.checkpoints_dir, self.name)
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_style', 'G_content']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['C', 'E', 'G']
        self.visual_names = ['A', 'B', 'C', 'R']
        self.input_nc = 3
        self.output_nc = 3
        self.ndf = 64 #number of filters in the first layer of discriminator
        self.ngf = 64

        use_sigmoid = False
        # define networks

        self.netCA = networks.define_channel_attention(self.gpu_ids)

        self.netKA = networks.define_kernel_attention(self.gpu_ids)

        self.netSA = networks.define_spatial_attention(self.gpu_ids)

        self.netC = networks.define_Convolution(self.gpu_ids)

        self.netKC = networks.define_K_Convolution(self.gpu_ids)

        self.netVGG = networks.define_VGG()

        self.netE = networks.define_E(self.input_nc, self.ngf, self.gpu_ids)

        self.netG = networks.define_G(self.input_nc, self.output_nc, self.ngf, self.gpu_ids)


        self.criterionMSE = torch.nn.MSELoss()

        self.criterionL1 = torch.nn.L1Loss()


        # initialize optimizers

        self.schedulers = []
        self.optimizers = []

        self.optimizer_CA = torch.optim.Adam(self.netCA.parameters(),
                                            lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_KA = torch.optim.Adam(self.netKA.parameters(),
                                             lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_SA = torch.optim.Adam(self.netSA.parameters(),
                                             lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_C = torch.optim.Adam(self.netC.parameters(),
                                            lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_KC = torch.optim.Adam(self.netKC.parameters(),
                                            lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_E = torch.optim.Adam(self.netE.parameters(),
                                            lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=0.0002, betas=(0.5, 0.999))

        self.optimizers.append(self.optimizer_CA)
        self.optimizers.append(self.optimizer_KA)
        self.optimizers.append(self.optimizer_SA)
        self.optimizers.append(self.optimizer_C)
        self.optimizers.append(self.optimizer_KC)
        self.optimizers.append(self.optimizer_E)
        self.optimizers.append(self.optimizer_G)

        for optimizer in self.optimizers:
            self.schedulers.append(networks.get_scheduler(optimizer, lr_policy='lambda', epoch_count=1, niter=100, niter_decay=100, lr_decay_iters=50))

        if not self.isTrain or args.continue_train:
            self.load_networks(self.which_epoch)

        self.print_networks()


    def set_input(self, input):

        self.A = input['A'].cuda()
        self.B = input['B'].cuda()
        self.C = input['C'].cuda()

        self.real_A = input['real_A'].cuda()
        self.real_B = input['real_B'].cuda()
        self.real_C = input['real_C'].cuda()


    def forward(self):

        self.latent_A = self.netE(self.A)
        self.latent_B = self.netE(self.B)
        self.latent_C = self.netE(self.C)

        # self.kernel_attention = self.netKA(self.latent_B)

        # self.channel_attention = self.netCA(self.latent_B)
        #
        # self.latent_R = self.channel_attention * self.latent_A

        # self.latent_R = self.netKC(self.kernel_attention, self.latent_A)

        # self.latent_R = self.kernel_attention * self.latent_A

        self.latent_R = self.netC(self.latent_A)

        self.R = self.netG(self.latent_R)

        self.vgg_A = normalize_batch(self.A)
        self.vgg_B = normalize_batch(self.B)
        self.vgg_C = normalize_batch(self.C)
        self.vgg_R = normalize_batch(self.R)

        self.feature_A = self.netVGG(self.vgg_A)
        self.feature_B = self.netVGG(self.vgg_B)
        self.feature_C = self.netVGG(self.vgg_C)
        self.feature_R = self.netVGG(self.vgg_R)


    def backward_G(self):


        self.gram_R = gram_matrix(self.feature_R)

        self.gram_B = gram_matrix(self.feature_B)

        self.loss_G_style = self.criterionMSE(self.gram_R, self.gram_B)
        #
        self.loss_G_content = self.criterionMSE(self.feature_R, self.feature_A) * 1

        # self.loss_G_L1 = self.criterionL1(self.R, self.C)
        #
        self.loss_G = self.loss_G_style + self.loss_G_content

        self.loss_G.backward()


    def optimize_parameters(self):

        self.forward()

        self.optimizer_E.zero_grad()
        self.optimizer_G.zero_grad()
        self.optimizer_C.zero_grad()
        # self.optimizer_KC.zero_grad()
        self.backward_G()
        self.optimizer_E.step()
        self.optimizer_G.step()
        self.optimizer_C.step()
        # self.optimizer_KC.step()


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

    def update_learning_rate(self):

        for scheduler in self.schedulers:
            scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = getattr(self, 'loss_' + name)
        return errors_ret

    def save_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (which_epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)
                torch.save(net.state_dict(), save_path)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
