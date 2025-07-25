# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Simple fully convolutional neural network (FCN) implementations."""

import torch as T
import torch.nn as nn
from torch import Tensor
from torch.nn.modules import Module

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Module.__module__ = "torch.nn"


class FCN_larger_modified(Module):
    """A simple 5 layer FCN with leaky relus and 'same' padding."""

    def __init__(
        self,
        in_channels: int,
        classes: int,
        num_filters: int = 128,
        output_smooth: float = 1e-2,
        log_outputs: bool = True,
    ) -> None:
        """Initializes the 5 layer FCN model.

        Args:
            in_channels: Number of input channels that the model will expect
            classes: Number of filters in the final layer
            num_filters: Number of filters in each convolutional layer
        """
        super(FCN_larger_modified, self).__init__()  # type: ignore[no-untyped-call]

        conv1 = nn.modules.Conv2d(
            in_channels, num_filters, kernel_size=11, stride=1, padding=5
        )
        conv2 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=7, stride=1, padding=3
        )
        conv3 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=5, stride=1, padding=2
        )

        self.backbone = nn.modules.Sequential(
            conv1,
            nn.modules.LeakyReLU(inplace=True),
            conv2,
            nn.modules.LeakyReLU(inplace=True),
            conv3,
        )

        self.last = nn.modules.Conv2d(
            num_filters, classes, kernel_size=1, stride=1, padding=0
        )

        self.output_smooth = output_smooth
        self.log_outputs = log_outputs

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model."""
        x = self.backbone(x)
        # add smoothing
        x = self.last(x).softmax(1) + self.output_smooth
        # renormalize and log
        x = nn.functional.normalize(x, p=1, dim=1)
        if self.log_outputs:
            x = x.log()

        return x


class FCN_modified(Module):
    """A simple 5 layer FCN with leaky relus and 'same' padding."""

    def __init__(
        self,
        in_channels: int,
        classes: int,
        num_filters: int = 64,
        output_smooth: float = 1e-2,
        log_outputs: bool = True,
    ) -> None:
        """Initializes the 5 layer FCN model.

        Args:
            in_channels: Number of input channels that the model will expect
            classes: Number of filters in the final layer
            num_filters: Number of filters in each convolutional layer
        """
        super(FCN_modified, self).__init__()  # type: ignore[no-untyped-call]

        conv1 = nn.modules.Conv2d(
            in_channels, num_filters, kernel_size=3, stride=1, padding=1
        )
        conv2 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )
        conv3 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )
        conv4 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )
        conv5 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )

        self.backbone = nn.modules.Sequential(
            conv1,
            nn.modules.LeakyReLU(inplace=True),
            conv2,
            nn.modules.LeakyReLU(inplace=True),
            conv3,
            nn.modules.LeakyReLU(inplace=True),
            conv4,
            nn.modules.LeakyReLU(inplace=True),
            conv5,
            nn.modules.LeakyReLU(inplace=True),
        )

        self.last = nn.modules.Conv2d(
            num_filters, classes, kernel_size=1, stride=1, padding=0
        )

        self.output_smooth = output_smooth
        self.log_outputs = log_outputs

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model."""
        x = self.backbone(x)
        # add smoothing
        x = self.last(x).softmax(1) + self.output_smooth
        # renormalize and log
        x = nn.functional.normalize(x, p=1, dim=1)
        if self.log_outputs:
            x = x.log()

        return x


class FCN(Module):
    """A simple 5 layer FCN with leaky relus and 'same' padding."""

    def __init__(self, in_channels: int, classes: int, num_filters: int = 64) -> None:
        """Initializes the 5 layer FCN model.

        Args:
            in_channels: Number of input channels that the model will expect
            classes: Number of filters in the final layer
            num_filters: Number of filters in each convolutional layer
        """
        super(FCN, self).__init__()

        conv1 = nn.modules.Conv2d(
            in_channels, num_filters, kernel_size=3, stride=1, padding=1
        )
        conv2 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )
        conv3 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )
        conv4 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )
        conv5 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )

        self.backbone = nn.modules.Sequential(
            conv1,
            nn.modules.LeakyReLU(inplace=True),
            conv2,
            nn.modules.LeakyReLU(inplace=True),
            conv3,
            nn.modules.LeakyReLU(inplace=True),
            conv4,
            nn.modules.LeakyReLU(inplace=True),
            conv5,
            nn.modules.LeakyReLU(inplace=True),
        )

        self.last = nn.modules.Conv2d(
            num_filters, classes, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model."""
        x = self.backbone(x)
        x = self.last(x)
        return x


class FCN_modified_batchnorm(Module):
    """A simple 5 layer FCN with leaky relus and 'same' padding."""

    def __init__(
        self,
        in_channels: int,
        classes: int,
        num_filters: int = 64,
        output_smooth: float = 1e-2,
        log_outputs: bool = True,
    ) -> None:
        """Initializes the 5 layer FCN model.

        Args:
            in_channels: Number of input channels that the model will expect
            classes: Number of filters in the final layer
            num_filters: Number of filters in each convolutional layer
        """
        super(FCN_modified_batchnorm, self).__init__()  # type: ignore[no-untyped-call]

        conv1 = nn.modules.Conv2d(
            in_channels, num_filters, kernel_size=3, stride=1, padding=1
        )
        conv2 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )
        conv3 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )
        conv4 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )
        conv5 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )

        norm1 = nn.BatchNorm2d(num_filters)
        norm2 = nn.BatchNorm2d(num_filters)
        norm3 = nn.BatchNorm2d(num_filters)
        norm4 = nn.BatchNorm2d(num_filters)
        norm5 = nn.BatchNorm2d(num_filters)

        self.backbone = nn.modules.Sequential(
            conv1,
            norm1,
            nn.modules.LeakyReLU(inplace=True),
            conv2,
            norm2,
            nn.modules.LeakyReLU(inplace=True),
            conv3,
            norm3,
            nn.modules.LeakyReLU(inplace=True),
            conv4,
            norm4,
            nn.modules.LeakyReLU(inplace=True),
            conv5,
            norm5,
            nn.modules.LeakyReLU(inplace=True),
        )

        self.last = nn.modules.Conv2d(
            num_filters, classes, kernel_size=1, stride=1, padding=0
        )

        self.output_smooth = output_smooth
        self.log_outputs = log_outputs

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model."""
        x = self.backbone(x)
        # add smoothing
        x = self.last(x).softmax(1) + self.output_smooth
        # renormalize and log
        x = nn.functional.normalize(x, p=1, dim=1)
        if self.log_outputs:
            x = x.log()

        return x
