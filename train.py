import time
from models.style_model import StyleModel

from util.visualizer import Visualizer
from data.train_dataset import TrainDataset
import torch

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')

parser.add_argument('--which_epoch', type=str, default='200', help='which epoch to load? set to latest to use latest cached model')
parser.add_argument('--phase', type=str, default='train')

if __name__ == '__main__':

    args = parser.parse_args()

    batch_size = 1
    total_iters = 0  # the total number of training iterations
    epoch_count = 1
    niter = 100
    niter_decay = 100
    print_freq = 30
    display_id = 1
    display_port = 8091
    display_freq = 30
    update_html_freq = 30 #'frequency of saving training results to html'
    save_latest_freq = 5000 # 'frequency of saving the latest results'
    save_epoch_freq = 20
    save_by_iter = False
    gpu_ids = [0]
    lr_decay = False
    lr_policy = 'lambda'
    lr_decay_iters = 50

    train_dataset = TrainDataset()
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)

    dataset_size = len(train_dataset)
    print('#training images = %d' % dataset_size)

    model = StyleModel(args)
    print("model [%s] was created" % (model.name))

    visualizer = Visualizer(display_id, display_port, args)
    total_steps = 0

    for epoch in range(epoch_count, niter + niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(train_dataloader):
            print('epoch %d, data %d' %(epoch, i * batch_size))
            iter_start_time = time.time()

            # for 15 steps, print loss, save images, display images, display losses
            if total_steps % print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += batch_size
            epoch_iter += batch_size
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % display_freq == 0:
                save_result = total_steps % update_html_freq == 0 # True or False
                visualizer.display_current_results(model.get_current_visuals(), epoch, epoch_iter, save_result)

            if total_steps % print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                if display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_steps % save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')

            iter_data_time = time.time()
        if epoch % save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, niter + niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
