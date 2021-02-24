import torch


def cal_center(n_classes=10, dim=256, c=10):

    mean_logits = torch.zeros(n_classes, dim)
    mean_logits[0, 0] = 1
    for k in range(1, n_classes):
        for j in range(k):
            mean_logits[k, j] = -(1 / (n_classes - 1) + torch.dot(mean_logits[k, :], mean_logits[j, :])) / mean_logits[j, j]
        mean_logits[k, k] = torch.sqrt(torch.abs(1 - torch.norm(mean_logits[k, :]).pow(2)))
    mean_logits = mean_logits * c

    return mean_logits
