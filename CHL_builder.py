

import torch
import torch.nn as nn
from UNet import UnetGenerator
import torchvision.models as models
from discriminator import Discriminator


class CHL(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, dim=64, pred_dim=64):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(CHL, self).__init__()

        # create the encoder

        self.encoder = Discriminator()


        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view


        z1 = (self.encoder(x1))  # NxC
        z2 = (self.encoder(x2))  # NxC


        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach()
