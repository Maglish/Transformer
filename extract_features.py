import torch
from PIL import Image
from vgg19 import Vgg19
from torchvision import transforms
import matplotlib.pyplot as plt
from util.util import calc_mean_std
import numpy as np

def load_image(filename, size=None, scale=None):

    img = Image.open(filename)

    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)

    return img


def gram_matrix(input):
    b, c, h, w = input.size()
    F = input.view(b, c, h * w)
    G = torch.bmm(F, F.transpose(1, 2))
    G.div_(h * w * b * c)
    return G

def preprocess_image(image_path):

    img = load_image(image_path)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406),
                              (0.229, 0.224, 0.225))])

    img = transform(img)
    return img

vgg1 = Vgg19(requires_grad=False)

vgg1.cuda()

image_path = 'result.jpg'

style_path = 'style.jpg'

photo_path = 'photo.jpg'

style_img = preprocess_image(style_path)

content_img = preprocess_image(photo_path)

# img = preprocess_image(image_path)

style_img = style_img.unsqueeze(0)

style_img = style_img.cuda()

style_conv3_1 = vgg1(style_img)


content_img = content_img.unsqueeze(0)

content_img = content_img.cuda()

content_conv3_1 = vgg1(content_img)


style_mean, style_std = calc_mean_std(style_conv3_1)

content_mean, content_std = calc_mean_std(content_conv3_1)

size = content_conv3_1.size()

normalized_feat = (content_conv3_1 - content_mean.expand(size)) / content_std.expand(size)

new_feat = normalized_feat * style_std.expand(size) + style_mean.expand(size)

conv3_1 = style_conv3_1.cpu().data.numpy()

# new_feat = new_feat.cpu().data.numpy()

new_G = gram_matrix(new_feat)

new_G = new_G.cpu().data.numpy()[0, :, :]

G = gram_matrix(style_conv3_1)

G = G.cpu().data.numpy()[0, :, :]

channels = conv3_1.shape[1]

# for channel in np.arange(channels):
#
#     featuremap = conv3_1[0, channel, :, :]
#
#     plt.figure()
#     plt.imshow(featuremap)
#     plt.savefig('photo_featuremap/%d_image.png' % channel)
#     # plt.show()

plt.figure()
plt.imshow(new_G)
plt.savefig('normalized_gram.png')


