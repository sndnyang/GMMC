import torch.nn as nn
from models import wideresnet
import models


class F(nn.Module):
    """
    In MMC, we only need to extract feature without last fully connected layer
    """
    def __init__(self, depth=28, width=2, norm=None, dropout_rate=0.0):
        super(F, self).__init__()
        self.f = wideresnet.Wide_ResNet(depth, width, norm=norm, dropout_rate=dropout_rate)
        # self.class_output = nn.Linear(self.f.last_dim, n_classes)

    def feature(self, x):
        return self.f(x)

    # def forward(self, x, y=None):
    #     penult_z = self.f(x)
    #     output = self.class_output(penult_z).squeeze()
    #     return output
