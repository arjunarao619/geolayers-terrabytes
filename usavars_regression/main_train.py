#!/usr/bin/env python3
import argparse
import time
import numpy as np
import h5py
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

import torchvision.transforms as T
import torchvision.models as models

import wandb

# ====================
# HDF5 Dataset
# ====================
class HDF5Dataset(Dataset):
    def __init__(self, h5_file, transform=None):
        """
        h5_file: path to HDF5 file containing:
            - 'images': shape (N, 7, H, W)
            - 'labels': shape (N,)
            - 'centroids': shape (N, 2) (optional)
        transform: a torchvision transform to apply on the image.
        """
        self.h5_file = h5_file
        self.transform = transform
        self.f = h5py.File(self.h5_file, 'r')
        self.images = self.f['images']
        self.labels = self.f['labels']
        if 'centroids' in self.f:
            self.centroids = self.f['centroids']
        else:
            self.centroids = None

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx]   # shape: (7, H, W)
        label = self.labels[idx]
        img = torch.tensor(img, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        if self.transform:
            img = self.transform(img)
        return {'image': img, 'labels': label}

# ====================
# Custom Transform: MixedChannelNormalize
# ====================
class MixedChannelNormalize:
    def __call__(self, tensor):
        """
        Assume tensor shape is (7, H, W) where channels 0-3 are in [0,255] and channels 4-6 are already in [0,1].
        Scale channels 0-3 to [0,1].
        """
        tensor[:4] = tensor[:4].float() / 255.0
        return tensor

# ====================
# Helper Functions
# ====================
def compute_mean_std(dataset, batch_size=16, num_workers=4):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    n_channels = None
    total_sum = None
    total_sum_sq = None
    total_pixels = 0
    for batch in loader:
        images = batch['image']  # (B, C, H, W)
        if n_channels is None:
            n_channels = images.size(1)
            total_sum = torch.zeros(n_channels)
            total_sum_sq = torch.zeros(n_channels)
        b, c, h, w = images.shape
        total_pixels += b * h * w
        total_sum += images.sum(dim=[0,2,3])
        total_sum_sq += (images ** 2).sum(dim=[0,2,3])
    mean = total_sum / total_pixels
    std = torch.sqrt((total_sum_sq / total_pixels) - mean**2)
    return mean, std

def compute_r2(outputs, targets):
    targets = targets.float()
    ss_res = torch.sum((targets - outputs) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    if ss_tot == 0:
        return 0.0
    r2 = 1 - (ss_res / ss_tot)
    return r2.item()

# ====================
# Custom Transform Wrapper (if needed)
# ====================
class TransformWrapper:
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    def __getitem__(self, index):
        sample = self.dataset[index]
        image = sample['image']
        label = sample['labels']
        return self.transform(image), label
    def __len__(self):
        return len(self.dataset)

class ChannelSelector:
    def __init__(self, num_channels):
        self.num_channels = num_channels
    def __call__(self, tensor):
        # Assume tensor shape is (7, H, W); select the first num_channels.
        return tensor[:self.num_channels]

class ScaleExtraChannels:
    def __call__(self, tensor):
        """
        If the input tensor has 7 channels (shape: [7, H, W]), multiply channels 4-6
        (indices 4,5,6) by 255 so that they are scaled from [0,1] to [0,255].
        """
        if tensor.shape[0] == 7:
            tensor[4:7] = tensor[4:7] * 255.0
        return tensor
# ====================
# Dataset Subsetting (only for training)
# ====================
def get_subset(dataset, fraction):
    N = len(dataset)
    subset_size = max(1, int(N * fraction))
    indices = random.sample(range(N), subset_size)
    return Subset(dataset, indices)

# ====================
# Main Training and Testing Function
# ====================
def main(args):
    # Set seeds for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    wandb.init(project="usavars-recaled-treecover-3-mod", config=vars(args))
    
    # --------------------
    # Dataset Preparation
    # --------------------
    train_full = HDF5Dataset(args.train_h5)
    val_full   = HDF5Dataset(args.val_h5)
    test_full  = HDF5Dataset(args.test_h5)
    
    # Subset only the training dataset.
    train_ds = get_subset(train_full, args.subset)
    # For validation and testing, use the full datasets.
    val_ds = val_full
    test_ds = test_full
    
    # For training:
    if args.input_channels == 7:
        base_transform = T.Compose([
            ChannelSelector(7),
            ScaleExtraChannels(),  # scale extra channels to 0-255
            T.RandomCrop((args.img_size, args.img_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip()
        ])
    elif args.input_channels == 4:  # using 4 channels
        base_transform = T.Compose([
            ChannelSelector(4),
            T.RandomCrop((args.img_size, args.img_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip()
        ])
    else:
        base_transform = T.Compose([
            ChannelSelector(3),
            T.RandomCrop((args.img_size, args.img_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip()
        ])

    # For evaluation (validation and test):
    if args.input_channels == 7:
        val_transform = T.Compose([
            ChannelSelector(7),
            ScaleExtraChannels(),  # scale extra channels to 0-255
            T.CenterCrop((args.img_size, args.img_size))
        ])
    elif args.input_channels == 4:
        val_transform = T.Compose([
            ChannelSelector(4),
            T.CenterCrop((args.img_size, args.img_size))
        ])
    else:
        val_transform = T.Compose([
            ChannelSelector(3),
            T.CenterCrop((args.img_size, args.img_size))
        ])

    
    train_ds = TransformWrapper(train_ds, base_transform)
    val_ds   = TransformWrapper(val_ds, val_transform)
    test_ds  = TransformWrapper(test_ds, val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    
    # --------------------
    # Model Setup
    # --------------------
    '''
    Modify this for larger ResNet Variants
    '''
    model = models.resnet50(pretrained=False)
    old_conv = model.conv1  # originally expects 3 channels.
    if args.input_channels == 3:
        # no change needed: original conv1 already handles 3 channels
        pass
    elif args.input_channels == 4:
        model.conv1 = nn.Conv2d(4, old_conv.out_channels,
                                kernel_size=old_conv.kernel_size,
                                stride=old_conv.stride,
                                padding=old_conv.padding,
                                bias=old_conv.bias is not None)
        with torch.no_grad():
            model.conv1.weight[:, :3] = old_conv.weight
            model.conv1.weight[:, 3:] = old_conv.weight.mean(dim=1, keepdim=True)
    elif args.input_channels == 7:
        model.conv1 = nn.Conv2d(7, old_conv.out_channels,
                                kernel_size=old_conv.kernel_size,
                                stride=old_conv.stride,
                                padding=old_conv.padding,
                                bias=old_conv.bias is not None)
        with torch.no_grad():
            model.conv1.weight[:, :3] = old_conv.weight
            model.conv1.weight[:, 3:] = old_conv.weight.mean(dim=1, keepdim=True)
    else:
        raise ValueError("input_channels must be either 3, 4 or 7")
    
    # Replace final fully connected layer for regression.
    model.fc = nn.Linear(model.fc.in_features, 1)
    
    if args.device == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)
    model = model.to(args.device)
    print("Model created.")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --------------------
    # Training Loop
    # --------------------
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        for batch_idx, sample in enumerate(train_loader):
            images = sample[0]
            targets = sample[1]
            # If using only 4 channels, slice out the first 4.
            # if args.input_channels == 4:
            #     images = images[:, :4, :, :]
            images = images.to(args.device)
            targets = targets.float().to(args.device)
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            r2 = compute_r2(outputs, targets)
            if (batch_idx + 1) % 1 == 0:
                print(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}, R²: {r2:.4f}")
                wandb.log({"batch_loss": loss.item(), "batch_r2": r2, "epoch": epoch+1})
        epoch_loss = running_loss / len(train_loader.dataset)
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {epoch_loss:.4f} Time: {elapsed:.2f}s")
        wandb.log({"epoch_loss": epoch_loss, "epoch": epoch+1})

        # --------------------
        # Validation Pass
        # --------------------
        model.eval()
        val_loss = 0.0
        val_r2_total = 0.0
        n_batches = 0
        with torch.no_grad():
            for sample in test_loader:
                images = sample[0]
                targets = sample[1]
                # if args.input_channels == 4:
                #     images = images[:, :4, :, :]
                images = images.to(args.device)
                targets = targets.float().to(args.device)
                outputs = model(images).squeeze()
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
                val_r2_total += compute_r2(outputs, targets)
                n_batches += 1
        val_loss /= len(val_loader.dataset)
        avg_val_r2 = val_r2_total / n_batches
        print(f"Validation Loss: {val_loss:.4f}, Validation R²: {avg_val_r2:.4f}")
        wandb.log({"val_loss": val_loss, "val_r2": avg_val_r2, "epoch": epoch+1})
    
    # --------------------
    # Testing Pass
    # --------------------
    model.eval()
    test_loss = 0.0
    test_r2_total = 0.0
    n_batches = 0
    with torch.no_grad():
        for sample in test_loader:
            images = sample[0]
            targets = sample[1]
            if args.input_channels == 4:
                images = images[:, :4, :, :]
            images = images.to(args.device)
            targets = targets.float().to(args.device)
            outputs = model(images).squeeze()
            loss = criterion(outputs, targets)
            test_loss += loss.item() * images.size(0)
            test_r2_total += compute_r2(outputs, targets)
            n_batches += 1
    test_loss /= len(test_loader.dataset)
    avg_test_r2 = test_r2_total / n_batches
    print(f"Test Loss: {test_loss:.4f}, Test R²: {avg_test_r2:.4f}")
    wandb.log({"test_loss": test_loss, "test_r2": avg_test_r2})
    
    # --------------------
    # Save the Model
    # --------------------
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")
    wandb.save(args.save_path)
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet50 for regression on USAVars HDF5 dataset")
    parser.add_argument("--img-size", type=int, default=256, help="Image size (height and width)")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers")
    parser.add_argument("--save-path", type=str, default="resnet50_usavars.pth", help="Path to save the trained model")
    parser.add_argument("--train-h5", type=str, default="usavars_train_20k.h5", help="Path to HDF5 training file")
    parser.add_argument("--val-h5", type=str, default="usavars_val.h5", help="Path to HDF5 validation file")
    parser.add_argument("--test-h5", type=str, default="usavars_test.h5", help="Path to HDF5 test file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    parser.add_argument("--input-channels", type=int, choices=[3,4,7], default=7, help="Number of input channels (4 or 7)")
    parser.add_argument("--subset", type=float, default=1.0, help="Fraction of the training dataset to use (e.g., 0.01 for 1%)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    main(args)
