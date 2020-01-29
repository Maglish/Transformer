import os
from options.test_options import TestOptions
from data.test_dataset import TestDataset
from models.test_model import TestModel
from util.visualizer import Visualizer
from data.test_dataset import TestDataset
from util import html
import time
import torch
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
parser.add_argument('--lambda_r', type=float, default=10, help='weight of reconstruction loss')
parser.add_argument('--lambda_s', type=float, default=10, help='weight of style loss')
parser.add_argument('--alpha', type=float, default=0.01, help='weight of content loss')
parser.add_argument('--lambda_d', type=float, default=1, help='weight of discriminative loss')

parser.add_argument('--which_epoch', type=str, default='100', help='which epoch to load? set to latest to use latest cached model')
parser.add_argument('--phase', type=str, default='test')

if __name__ == '__main__':
    args = parser.parse_args()
    nThreads = 1   # test code only supports nThreads = 1
    batchSize = 1  # test code only supports batchSize = 1
    serial_batches = True  # no shuffle
    no_flip = True  # no flip
    display_id = -1
    display_port = 8094
    # no visdom display

    test_dataset = TestDataset()
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batchSize,
        shuffle=False,
        num_workers=4)
    # name = 'G_GAN_%s_lambdar_%s_lambdas_%s_alpha_%s' % (args.lambda_d,args.lambda_r, args.lambda_s, args.alpha)
    name = 'StyleRetainer'
    model = TestModel(name, args.which_epoch)
    visualizer = Visualizer(display_id, display_port, args)

    # create saving folder
    folder_dir = os.path.join(args.results_dir, name, '%s_%s' % (args.phase, args.which_epoch))

    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)

    # test
    for i, data in enumerate(test_dataloader):
        start = time.time()
        model.set_input(data)
        model.test()
        end = time.time()
        print(end-start)
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()

        visualizer.save_images(folder_dir, visuals, img_path)

        print('%04d: process image... %s' % (i, img_path))
