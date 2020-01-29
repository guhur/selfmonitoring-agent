"""
Several models for Feature-wise linear modulations.
"""
from typing import Type, List
import torch
from torch import nn
import torch.nn.functional as F


class FiLMGenerator(nn.Module):
    """
    Generator for the scaling/additional parameters from FiLM
    """

    def __init__(self, context_size, num_resblocks, conv_hidden):
        """
        Args:
            context_size (int): size of the context vector
        """
        super().__init__()

        self.hidden_size = num_resblocks * conv_hidden

        self.out = nn.Sequential(
            nn.Linear(context_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size * 2),
        )

    def forward(self, context_vector):
        return self.out(context_vector)


class FiLMResidual(nn.Module):
    """ Residual part of a ResNet with the FiLM transformation """

    def forward(self, feat, res, gamma, beta):
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        feat = (1 + gamma) * feat + beta
        feat = F.relu(feat)
        feat = feat + res
        return feat


class FiLMedResBlock(nn.Module):
    def __init__(self, conv_hidden, with_batch_norm=True):
        super().__init__()
        self.with_batch_norm = with_batch_norm

        self.conv1 = nn.Conv2d(
            conv_hidden, conv_hidden, kernel_size=[3, 3], padding=1
        )
        self.nonlin = nn.ReLU()
        self.conv2 = nn.Conv2d(
            conv_hidden, conv_hidden, kernel_size=[3, 3], padding=1
        )
        self.bn = nn.BatchNorm2d(conv_hidden)
        self.filmres = FiLMResidual()

    def forward(self, input, gamma, beta):
        out1 = self.nonlin(self.conv1(input))
        out2 = self.conv2(out1)
        if self.with_batch_norm:
            out2 = self.bn(out2)

        out = self.filmres(out2, out1, gamma, beta)
        out = out.max(dim=1).values
        out = out.flatten(start_dim=1)
        return out


class FiLMedResBlocks(nn.Module):
    """
    Modulates multiple resblocks
    """

    def __init__(
        self,
        num_blocks: int = 4,
        conv_hidden: int = 128,
        with_batch_norm: bool = True,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.conv_hidden = conv_hidden

        self.resblocks: nn.Module = nn.ModuleList(
            FiLMedResBlock(self.conv_hidden, with_batch_norm=with_batch_norm)
            for _ in range(self.num_blocks)
        )

    def forward(self, features, film_parameters):
        film = film_parameters.chunk(self.num_blocks * 2, 1)

        for i, resblock in enumerate(self.resblocks):
            features = resblock(features, film[i * 2], film[i * 2 + 1])

        return features

class FiLMTail(nn.Module):

    def __init__(self, in_features: int, out_features:int ):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images' shape: batch_size x num_img x channels x height x width
        """
        # (batch_size x num_img x channels)
        images = self.avgpool(images).squeeze(3).squeeze(3)
        # (batch_size x num_img x out_features)
        images = self.fc(images)

        return images

        

