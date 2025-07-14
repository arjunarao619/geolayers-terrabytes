from typing import Any, Callable, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def center_crop(size: int = 512) -> Callable[[Dict[str, Tensor]], Dict[str, Tensor]]:
    """Returns a function to perform a center crop transform on a single sample.
    Args:
        size: output image size
    Returns:
        function to perform center crop
    """

    def center_crop_inner(sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        _, height, width = sample["image"].shape

        y1 = (height - size) // 2
        x1 = (width - size) // 2
        sample["image"] = sample["image"][:, y1 : y1 + size, x1 : x1 + size]
        sample["mask"] = sample["mask"][:, y1 : y1 + size, x1 : x1 + size]

        return sample

    return center_crop_inner


def nodata_check(size: int = 512) -> Callable[[Dict[str, Tensor]], Dict[str, Tensor]]:
    """Returns a function to check for nodata or mis-sized input.
    Args:
        size: output image size
    Returns:
        function to check for nodata values
    """

    def nodata_check_inner(sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        num_channels, height, width = sample["image"].shape

        if height < size or width < size:
            sample["image"] = torch.zeros((num_channels, size, size))
            sample["mask"] = torch.zeros((size, size))

        return sample

    return nodata_check_inner


def preprocess_image(sample: Dict[str, Any]) -> Dict[str, Any]:

    def preprocess(sample: Dict[str, Any]) -> Dict[str, Any]:
        onehot_encode_labels = False
        classes_keep = [1, 2, 3, 4, 6]
        num_nlcd_layers = 5
        ignore_index = len(classes_keep)
        n_classes = len(classes_keep)
        n_classes_with_nodata = len(classes_keep) + 1
        """Preprocesses a single sample."""
        # sample['image'] contains the weak inputs, sample['mask'] is the hr labelsß

        # normalize just the NLCD layers because they get stored as 0...255
        sample["image"] = sample["image"].float()
        sample["image"][:] = sample["image"][:] / 255.0
        # sample["image"][4:] = torch.where(sample["image"][4:] > 0, 1, 0) # Normalizing roads to be between 0 and 1
        # handle reindexing the labels
        reindex_map = dict(zip(classes_keep, np.arange(len(classes_keep))))
        # reindex shrub to tree for learning the prior
        tree_idx = 3  # tree idx is 3 when there are no zeros
        shrub_idx = 5
        reindex_map[shrub_idx] = tree_idx
        reindexed_mask = -1 * torch.ones(sample["mask"].shape)
        for old_idx, new_idx in reindex_map.items():
            reindexed_mask[sample["mask"] == old_idx] = new_idx

        reindexed_mask[reindexed_mask == -1] = ignore_index
        assert (reindexed_mask >= 0).all()

        sample["mask"] = reindexed_mask

        if onehot_encode_labels:
            sample["mask"] = (
                nn.functional.one_hot(
                    sample["mask"].to(torch.int64), num_classes=n_classes
                )
                .transpose(0, 2)
                .transpose(1, 2)
            )

        sample["mask"] = sample["mask"].squeeze().long()

        del sample["bounds"]

        return sample

    return preprocess(sample)


def pad_to(
    size: int = 512, image_value: int = 0, mask_value: int = 0
) -> Callable[[Dict[str, Tensor]], Dict[str, Tensor]]:
    """Returns a function to perform a padding transform on a single sample.
    Args:
        size: output image size
        image_value: value to pad image with
        mask_value: value to pad mask with
    Returns:
        function to perform padding
    """

    def pad_inner(sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        _, height, width = sample["image"].shape
        assert height <= size and width <= size

        height_pad = size - height
        width_pad = size - width

        # See https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        # for a description of the format of the padding tuple
        sample["image"] = F.pad(
            sample["image"],
            (0, width_pad, 0, height_pad),
            mode="constant",
            value=image_value,
        )
        sample["mask"] = F.pad(
            sample["mask"],
            (0, width_pad, 0, height_pad),
            mode="constant",
            value=mask_value,
        )
        return sample

    return pad_inner
