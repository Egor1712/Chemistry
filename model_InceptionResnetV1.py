import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image
import os


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False
        ) # verify bias false
        self.bn = nn.BatchNorm2d(
            out_planes,
            eps=0.001, # value found in tensorflow
            momentum=0.1, # default pytorch value
            affine=True
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super().__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(256, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(96, 256, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super().__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(896, 128, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(896, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 128, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(128, 128, kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.conv2d = nn.Conv2d(256, 896, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block8(nn.Module):

    def __init__(self, scale=1.0, noReLU=False):
        super().__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = BasicConv2d(1792, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1792, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(1,3), stride=1, padding=(0,1)),
            BasicConv2d(192, 192, kernel_size=(3,1), stride=1, padding=(1,0))
        )

        self.conv2d = nn.Conv2d(384, 1792, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class Mixed_6a(nn.Module):

    def __init__(self):
        super().__init__()

        self.branch0 = BasicConv2d(256, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(256, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1),
            BasicConv2d(192, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2, return_indices=True)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2, indices = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out, indices


class Mixed_7a(nn.Module):

    def __init__(self):
        super().__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(3, stride=2, return_indices=True)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3, indices = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out, indices


class InceptionResnetV1(nn.Module):
    """Inception Resnet V1 model with optional loading of pretrained weights.

    Model parameters can be loaded based on pretraining on the VGGFace2 or CASIA-Webface
    datasets. Pretrained state_dicts are automatically downloaded on model instantiation if
    requested and cached in the torch cache. Subsequent instantiations use the cache rather than
    redownloading.

    Keyword Arguments:
        pretrained {str} -- Optional pretraining dataset. Either 'vggface2' or 'casia-webface'.
            (default: {None})
        classify {bool} -- Whether the model should output classification probabilities or feature
            embeddings. (default: {False})
        num_classes {int} -- Number of output classes. If 'pretrained' is set and num_classes not
            equal to that used for the pretrained model, the final linear layer will be randomly
            initialized. (default: {None})
        dropout_prob {float} -- Dropout probability. (default: {0.6})
    """
    def __init__(self, dropout_prob=0.6, device=None):
        super().__init__()

        # Define layers
        # self.conv2d_1a = BasicConv2d(4, 32, kernel_size=3, stride=2)
        # self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        # self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        # self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        # self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        # self.conv2d_4b = BasicConv2d(192, 256, kernel_size=3, stride=2)
        self.conv2d_1a = BasicConv2d(4, 32, kernel_size=3, stride=2, padding=0)
        self.conv2d_2a = BasicConv2d(32, 64, kernel_size=1, stride=1, padding=2)
        self.conv2d_2b = BasicConv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.conv2d_3a = BasicConv2d(192, 256, kernel_size=3, stride=1, padding=1)

        self.repeat_1 = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
        )
        self.mixed_6a = Mixed_6a()
        self.repeat_2 = nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
        )
        self.mixed_7a = Mixed_7a()
        self.repeat_3 = nn.Sequential(
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
        )
        self.block8 = Block8(noReLU=True)
        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)

        self.device = torch.device('cpu')
        if device is not None:
            self.device = device
            self.to(device)

    def forward(self, x):
        """Calculate embeddings or logits given a batch of input image tensors.

        Arguments:
            x {torch.tensor} -- Batch of image tensors representing faces.

        Returns:
            torch.tensor -- Batch of embedding vectors or multinomial logits.
        """
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.conv2d_3a(x)
        x = self.repeat_1(x)
        x, indices_6 = self.mixed_6a(x)
        x = self.repeat_2(x)
        x, indices_7 = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        return x, indices_6, indices_7


class BasicConvTranspose2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False
        ) # verify bias false
        self.bn = nn.BatchNorm2d(
            out_planes,
            eps=0.001, # value found in tensorflow
            momentum=0.1, # default pytorch value
            affine=True
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Block35_D(nn.Module):

    def __init__(self, operator, scale=1.0):
        super().__init__()

        self.scale = scale

        self.branch0 = operator(256, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            operator(256, 32, kernel_size=1, stride=1),
            operator(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            operator(256, 32, kernel_size=1, stride=1),
            operator(32, 32, kernel_size=3, stride=1, padding=1),
            operator(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.ConvTranspose2d(96, 256, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block17_D(nn.Module):

    def __init__(self, operator, scale=1.0):
        super().__init__()

        self.scale = scale

        self.branch0 = operator(896, 128, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            operator(896, 128, kernel_size=1, stride=1),
            operator(128, 128, kernel_size=(1,7), stride=1, padding=(0,3)),
            operator(128, 128, kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.conv2d = nn.ConvTranspose2d(256, 896, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block8_D(nn.Module):

    def __init__(self, operator, scale=1.0, noReLU=False):
        super().__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = operator(1792, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            operator(1792, 192, kernel_size=1, stride=1),
            operator(192, 192, kernel_size=(1,3), stride=1, padding=(0,1)),
            operator(192, 192, kernel_size=(3,1), stride=1, padding=(1,0))
        )

        self.conv2d = nn.ConvTranspose2d(384, 1792, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class Mixed_6d(nn.Module):

    def __init__(self):
        super().__init__()

        self.branch1_1 = BasicConvTranspose2d(896, 256, kernel_size=1, stride=1)
        self.branch1_2 = nn.MaxUnpool2d(3, stride=2)

    def forward(self, x, indices):
        x = self.branch1_1(x)
        x = self.branch1_2(x, indices)
        return x


class Mixed_7d(nn.Module):

    def __init__(self):
        super().__init__()

        self.branch0 = nn.Sequential(
            BasicConvTranspose2d(1792, 256, kernel_size=1, stride=1),
            BasicConvTranspose2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConvTranspose2d(1792, 256, kernel_size=1, stride=1),
            BasicConvTranspose2d(256, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            BasicConvTranspose2d(1792, 256, kernel_size=1, stride=1),
            BasicConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConvTranspose2d(256, 256, kernel_size=3, stride=2)
        )

        self.branch3_1 = BasicConvTranspose2d(1792, 896, kernel_size=1, stride=1)
        self.branch3_2 = nn.MaxUnpool2d(3, stride=2)

    def forward(self, x, indices):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3_1(x)
        x3 = self.branch3_2(x3, indices)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.deconv6 = nn.Sequential(nn.ConvTranspose2d(1792, 1792, kernel_size=8, stride=1, padding=0), nn.BatchNorm2d(1792), nn.ReLU())

        self.deconv5 = nn.Sequential(
                    Block8_D(BasicConvTranspose2d, scale=0.17),
                    Block8_D(BasicConvTranspose2d, scale=0.17),
                    Block8_D(BasicConvTranspose2d, scale=0.17),
                    Block8_D(BasicConvTranspose2d, scale=0.17),
                    Block8_D(BasicConvTranspose2d, scale=0.17),
            )

        self.deconv4 = Mixed_7d()

        self.deconv3 = nn.Sequential(
                    Block17_D(BasicConvTranspose2d, scale=0.17),
                    Block17_D(BasicConvTranspose2d, scale=0.17),
                    Block17_D(BasicConvTranspose2d, scale=0.17),
                    Block17_D(BasicConvTranspose2d, scale=0.17),
                    Block17_D(BasicConvTranspose2d, scale=0.17),
                    Block17_D(BasicConvTranspose2d, scale=0.17),
                    Block17_D(BasicConvTranspose2d, scale=0.17),
                    Block17_D(BasicConvTranspose2d, scale=0.17),
                    Block17_D(BasicConvTranspose2d, scale=0.17),
                    Block17_D(BasicConvTranspose2d, scale=0.17),
            )

        self.deconv2 = Mixed_6d()

        self.deconv1 = nn.Sequential(
                    Block35_D(BasicConvTranspose2d, scale=0.17),
                    Block35_D(BasicConvTranspose2d, scale=0.17),
                    Block35_D(BasicConvTranspose2d, scale=0.17),
                    Block35_D(BasicConvTranspose2d, scale=0.17),
                    Block35_D(BasicConvTranspose2d, scale=0.17),
            )

        self.deconv0 = nn.Sequential(
            nn.ConvTranspose2d(256, 192, kernel_size=3, stride=1, padding=2),
            nn.ConvTranspose2d(192, 64, kernel_size=3, stride=2, padding=2),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=1, padding=0),
            nn.ConvTranspose2d(32, 16, kernel_size=1, stride=1, padding=0),
            nn.ConvTranspose2d(16, 4, kernel_size=3, stride=1, padding=1),
            )

    def forward(self, x, indices_6, indices_7):
        x = self.deconv6(x)
        x = self.deconv5(x)
        x = self.deconv4(x, indices_7)
        x = self.deconv3(x)
        x = self.deconv2(x, indices_6)
        x = self.deconv1(x)
        x = self.deconv0(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.encoder = InceptionResnetV1(device=device)
        self.decoder = Decoder()

    def forward(self, x):
        z, indices_6, indices_7 = self.encoder(x)
        out = self.decoder(z, indices_6, indices_7)
        return out, z
