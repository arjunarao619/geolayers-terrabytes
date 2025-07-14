# models/ModelFactory.py

import timm
import torch

from models.fcn import FCN, FCN_larger_modified, FCN_modified
from models.resnet import ResNet50, ResNet101
from models.unet import UNet


class ModelFactory:
    """
    A factory class to retrieve and instantiate different models based on configuration.
    """

    def __init__(
        self,
        model_name: str,
        input_channels: int,
        num_classes: int,
        pretrained: bool = True,
    ):
        """
        Initializes the ModelFactory with the desired model configuration.

        Args:
            model_name (str): The name of the model to instantiate.
            input_channels (int): Number of input channels.
            num_classes (int): Number of output classes.
            pretrained (bool): Whether to use pretrained weights (if applicable).
        """
        self.model_name = model_name.lower()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.pretrained = pretrained

    def get_model(self):
        """
        Retrieves the model instance based on the specified configuration.

        Returns:
            torch.nn.Module: The instantiated model.

        Raises:
            ValueError: If the specified model_name is unsupported.
        """
        if self.model_name == "resnet50":
            return ResNet50(channels=self.input_channels, num_classes=self.num_classes)
        elif self.model_name == "resnet101":
            return ResNet101(channels=self.input_channels, num_classes=self.num_classes)
        elif self.model_name == "unet":
            return UNet(
                n_channels=self.input_channels,
                n_classes=self.num_classes,
                bilinear=True,
            )
        elif self.model_name == "fcn":
            # Choose the appropriate FCN variant as needed
            return FCN(in_channels=self.input_channels, classes=self.num_classes)
            # Alternatively, use a modified FCN:
            # return FCN_modified(in_channels=self.input_channels, classes=self.num_classes)
        elif self.model_name == "resnet18":
            return timm.create_model(
                "resnet18",
                pretrained=self.pretrained,
                in_chans=self.input_channels,
                num_classes=self.num_classes,
            )
        elif self.model_name == "resnet152":
            return timm.create_model(
                "resnet152",
                pretrained=self.pretrained,
                in_chans=self.input_channels,
                num_classes=self.num_classes,
            )
        elif self.model_name == "efficientnet":
            return timm.create_model(
                "efficientnet_lite0",
                pretrained=self.pretrained,
                in_chans=self.input_channels,
                num_classes=self.num_classes,
            )
        elif self.model_name == "mobilenet":
            return timm.create_model(
                "mobilenetv3_small_100",
                pretrained=self.pretrained,
                in_chans=self.input_channels,
                num_classes=self.num_classes,
            )
        elif self.model_name == "cspnet":
            return timm.create_model(
                "cspresnet50",
                pretrained=self.pretrained,
                in_chans=self.input_channels,
                num_classes=self.num_classes,
            )
        elif self.model_name == "densenet":
            return timm.create_model(
                "densenet121",
                pretrained=self.pretrained,
                in_chans=self.input_channels,
                num_classes=self.num_classes,
            )
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")
