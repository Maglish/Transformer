import torch
import torch.nn as nn

def cal_mean_std(feat, eps=1e-5):

    size = feat.size()
    assert (len(size) == 4)
    N,C = size[:2]

    feat_var = feat.view(N,C,-1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N,C,-1).mean(dim=2).view(N,C,1,1)
    return feat_mean, feat_std