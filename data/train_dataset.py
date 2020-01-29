import torch
import os
import torchvision.transforms as transforms
from PIL import Image


def is_image_file(filename):
    IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    ]
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []

    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

def get_transform():
    transform_list = []
    load_size = 256
    crop_size = 256
    #resize
    osize = [load_size, load_size]
    transform_list.append(transforms.Resize(osize, Image.BICUBIC))
    #crop
    transform_list.append(transforms.RandomCrop(crop_size))

    # transform_list.append(transforms.RandomHorizontalFlip())

    #tensor
    transform_list += [transforms.ToTensor()]

    transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)


def get_vggtransform():

    transform_list = []
    transform_list += [transforms.ToTensor()]

    return transforms.Compose(transform_list)


class TrainDataset(torch.utils.data.Dataset):

    def __init__(self):
        dataroot = './datasets'
        dir_A = os.path.join(dataroot + '/photo')
        dir_B = os.path.join(dataroot + '/style'+'/style.jpg')
        dir_C = os.path.join(dataroot + '/results')
        self.A_paths = make_dataset(dir_A)
        self.B_paths = dir_B
        self.C_paths = make_dataset(dir_C)
        self.no_flip = False
        self.transform = get_transform() #transform list
        self.vgg_transform = get_vggtransform()  # no 0.5 normalization


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        A = self.transform(A_img)
        real_A = self.vgg_transform(A_img)

        B_path = self.B_paths
        B_img = Image.open(B_path).convert('RGB')
        B = self.transform(B_img)
        real_B = self.vgg_transform(B_img)

        C_path = self.C_paths[index]
        C_img = Image.open(C_path).convert('RGB')
        C = self.transform(C_img)
        real_C = self.vgg_transform(C_img)

        return {'A': A, 'A_paths': A_path, 'real_A': real_A,
                'B': B, 'B_paths': B_path, 'real_B': real_B,
                'C': C, 'C_paths': C_path, 'real_C': real_C}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)

    def name(self):

        return 'TrainDataset'






