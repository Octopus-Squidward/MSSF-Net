import torch
import torch.nn as nn


def l12_norm(inputs):
    out = torch.mean(torch.sum(torch.sqrt(torch.abs(inputs)), dim=1))
    return out


class MinVolumn(nn.Module):
    def __init__(self, band, num_classes, delta):
        super(MinVolumn, self).__init__()
        self.band = band
        self.delta = delta
        self.num_classes = num_classes

    def __call__(self, edm):
        edm_result = torch.reshape(edm, (self.band, self.num_classes))
        edm_mean = edm_result.mean(dim=1, keepdim=True)
        loss = self.delta * ((edm_result - edm_mean) ** 2).sum() / self.band / self.num_classes
        return loss


class SparseLoss(nn.Module):
    def __init__(self, sparse_decay):
        super(SparseLoss, self).__init__()
        self.sparse_decay = sparse_decay

    def __call__(self, input):
        B, P, _, _ = input.shape
        input = input.reshape(B, P)
        loss = l12_norm(input)
        return self.sparse_decay * loss
