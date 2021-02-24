import numpy as np
import torch
import torch.nn as nn
from ExpUtils import wlog
print = wlog


class LDALoss(nn.Module):
    """
    LGMLoss whose covariance is fixed as Identity matrix
    """

    def __init__(self, num_classes, feat_dim, alpha, args=None):
        super(LDALoss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        #self.alpha = alpha

        # if 'method' in vars(args) and 'mmc' in args.method:
        #     self.centers = 0
        # else:
        #self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim), requires_grad=False)

    def forward(self, feat, y=None):
        diff = torch.unsqueeze(feat, dim=1) - torch.unsqueeze(self.centers, dim=0)
        diff = torch.mul(diff, diff)
        dist = torch.sum(diff, dim=-1)
        logits = -0.5 * dist
        return logits

    def P_X_y(self, feat, u_y=None, y=None):
        assert not (u_y is None and y is None)
        if y is not None:
            u_y = torch.index_select(self.centers, 0, y)
        diff = feat - u_y
        diff = torch.mul(diff, diff)
        dist = torch.sum(diff, dim=1)
        likelihood = torch.mean(0.5 * dist)
        return likelihood

    def P_X(self, feat, y=None, scale=1):
        # reshape [batch_size, k] -> [batch_size, k, classes]
        t = feat.view(feat.shape[0], -1, 1).repeat(1, 1, self.num_classes)
        k = feat.shape[1]
        # subtract
        m = t - self.centers.transpose(1, 0).view(1, k, self.num_classes)
        # ||feature - mu||^2_2
        norm_2 = m.pow(2).sum(1)
        # logits = norm_2 * -.5

        # batch_size, 10
        d_min = 0.5 * norm_2.min(dim=1, keepdims=True)[0]
        # p_x_and_y = torch.exp(-0.5 * norm_2 + d_min)
        p_x_and_y = torch.exp(-0.5 * norm_2 * scale)

        # batch_size,
        # C = self.num_classes
        # p_x = 1.0 / C * (2 * np.pi)**(-k / 2) * p_x_and_y.sum(1)
        if y is None:
            p_x = p_x_and_y.sum(1)
        else:
            p_x = torch.gather(p_x_and_y, 1, y[:, None])

        # p_X namely log sum exp(X)
        # log_sum_exp = torch.log(d_min[:, 0] - torch.log(p_x))
        log_sum_exp = torch.log(p_x)
        return log_sum_exp
