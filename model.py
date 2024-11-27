import warnings
import os
import torch.optim
import torch.nn as nn

import config_image_s1 as c
from CNGI import NGI



class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()

        self.model = NGI()

    def forward(self, x, rev=False):

        if not rev:
            out = self.model(x)

        else:
            out = self.model(x, rev=True)

        return out





def init_model(mod):
    for key, param in mod.named_parameters():
        split = key.split('.')
        if param.requires_grad:
            param.data = c.init_scale * torch.randn(param.data.shape).cuda()
            if split[-2] == 'conv5':
                param.data.fill_(0.)
