import torch.nn as nn
# from options import HiDDenConfiguration
from conv_bn_relu import ConvBNRelu
import config_image_s1 as c

class Discriminator(nn.Module):
    """
    Discriminator network. Receives an image and has to figure out whether it has a watermark inserted into it, or not.
    """
    def __init__(self, discriminator_blocks=c.discriminator_blocks, discriminator_channels=c.discriminator_channels):
        super(Discriminator, self).__init__()

        layers = [ConvBNRelu(3, discriminator_channels)]
        for _ in range(discriminator_blocks-1):
            layers.append(ConvBNRelu(discriminator_channels, discriminator_channels))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.before_linear = nn.Sequential(*layers)
        self.linear = nn.Linear(discriminator_channels, 1)

    def forward(self, image):
        X = self.before_linear(image)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        X = X.squeeze_(3).squeeze_(2)
        # X = self.linear(X)
        # X = torch.sigmoid(X)
        return X