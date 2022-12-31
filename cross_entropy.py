import torch.nn.functional as F


def cross_entropy(input, target):
    log_softmax = F.log_softmax(input)
    batch_size = input.shape[0]
    return -(target * log_softmax).sum() / batch_size
