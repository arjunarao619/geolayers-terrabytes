import argparse
import h5py
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import random
from PIL import Image

# ============================
# U-Net Model Definition
# ============================
class UNet(nn.Module):
    def __init__(self, in_channels=7, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder: Downsampling path
        for feature in features:
            self.encoder.append(self.double_conv(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = self.double_conv(features[-1], features[-1]*2)
        
        # Decoder: Upsampling path
        reversed_features = features[::-1]
        self.upconvs = nn.ModuleList()
        self.decoder = nn.ModuleList()
        current_in_channels = features[-1]*2  # Start with bottleneck channels
        
        for feature in reversed_features:
            self.upconvs.append(
                nn.ConvTranspose2d(current_in_channels, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(self.double_conv(current_in_channels, feature))
            current_in_channels = feature
        
        # Final Convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for idx in range(len(self.upconvs)):
            x = self.upconvs[idx](x)
            skip_connection = skip_connections[idx]
            if x.shape[2:] != skip_connection.shape[2:]:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx](x)
        
        return self.final_conv(x)
    
    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

# ============================
# Custom Dataset Class for HDF5 Data
# ============================
class HDF5Dataset(Dataset):
    def __init__(self, hdf5_path, split='train', transform=None, channels="all"):
        """
        Custom Dataset for loading images and masks from an HDF5 file.
        The images are assumed to have shape (N, 7, H, W). When channels="rgb",
        only the first 3 channels are used; when channels="all", all 7 channels are used.
        Channel-wise normalization is applied using pre-computed statistics.
        """
        self.hdf5_path = hdf5_path
        self.split = split
        self.transform = transform
        self.channels = channels.lower()  # either "rgb" or "all"

        self.h5_file = h5py.File(self.hdf5_path, 'r')
        if 'images' in self.h5_file[self.split] and 'masks' in self.h5_file[self.split]:
            self.images = self.h5_file[self.split]['images']
            self.masks = self.h5_file[self.split]['masks']
        else:
            raise KeyError("HDF5 file must contain 'images' and 'masks' datasets under the specified split.")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load the full 7-channel image (shape: (7, H, W))
        image = self.images[idx].astype(np.float32)
        # Depending on training mode, select channels.
        if self.channels == "rgb":
            # Select only the first 3 channels.
            image = image[:3]
            channel_means = np.array([61.18, 75.86, 81.15], dtype=np.float32)
            channel_stds  = np.array([48.14, 31.38, 29.15], dtype=np.float32)
        else:
            channel_means = np.array([61.18, 75.86, 81.15, 0.1545, 0.09379, 0.09237, 271.83], dtype=np.float32)
            channel_stds  = np.array([48.14, 31.38, 29.15, 0.20566, 0.13182, 0.13612, 287.20], dtype=np.float32)
        
        # Normalize each channel: (x - mean) / std.
        for c in range(image.shape[0]):
            image[c] = (image[c] - channel_means[c]) / channel_stds[c]
        
        # Load mask; assume mask is stored as (H, W, 3) and average if necessary.
        mask = self.masks[idx]
        if mask.ndim == 3:
            mask = np.mean(mask, axis=2)
        mask = mask.astype(np.float32)
        
        if self.transform:
            image = self.transform(image)
        # Convert image and mask to tensors.
        image_tensor = torch.from_numpy(image)  # shape: (C, H, W)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # shape: (1, H, W)
        mask_tensor = (mask_tensor > 0).float()  # Binarize mask
        return image_tensor, mask_tensor
    
    def close(self):
        self.h5_file.close()

# ============================
# Evaluation Metrics
# ============================
def dice_coefficient(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean().item()

def iou_score(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()

# ============================
# Training and Validation Functions
# ============================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    for inputs, masks in tqdm(dataloader, desc="Training", leave=False):
        inputs = inputs.to(device)
        masks = masks.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_dice += dice_coefficient(outputs, masks) * inputs.size(0)
        running_iou += iou_score(outputs, masks) * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_dice = running_dice / len(dataloader.dataset)
    epoch_iou = running_iou / len(dataloader.dataset)
    return epoch_loss, epoch_dice, epoch_iou

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    with torch.no_grad():
        for inputs, masks in tqdm(dataloader, desc="Validation", leave=False):
            inputs = inputs.to(device)
            masks = masks.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, masks)

            running_loss += loss.item() * inputs.size(0)
            running_dice += dice_coefficient(outputs, masks) * inputs.size(0)
            running_iou += iou_score(outputs, masks) * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_dice = running_dice / len(dataloader.dataset)
    epoch_iou = running_iou / len(dataloader.dataset)
    return epoch_loss, epoch_dice, epoch_iou

def visualize_predictions(model, dataloader, device, channels_mode):
    model.eval()
    with torch.no_grad():
        for inputs, masks in dataloader:
            inputs = inputs.to(device)
            masks = masks.to(device)
            outputs = model(inputs)
            break

    for i in range(inputs.shape[0]):
        input_sample = inputs[i].cpu()  # shape: (C, H, W)
        mask_sample = masks[i].cpu().squeeze(0).numpy()
        pred_sample = torch.sigmoid(outputs[i].cpu()).squeeze(0)
        pred_sample = (pred_sample > 0.5).float().numpy()

        if channels_mode == "rgb":
            rgb_sample = input_sample[:3]
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
            rgb_sample = rgb_sample * std + mean
            rgb_np = rgb_sample.numpy().transpose(1,2,0)
            plt.figure(figsize=(12,4))
            plt.subplot(1,3,1)
            plt.imshow(rgb_np)
            plt.title("RGB (Denorm)")
            plt.axis('off')
        else:
            rgb_sample = input_sample[:3]
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
            rgb_sample = rgb_sample * std + mean
            rgb_np = rgb_sample.numpy().transpose(1,2,0)
            plt.figure(figsize=(12,4))
            plt.subplot(1,3,1)
            plt.imshow(rgb_np)
            plt.title("RGB (Denorm)")
            plt.axis('off')
        plt.subplot(1,3,2)
        plt.imshow(pred_sample, cmap='gray')
        plt.title("Prediction")
        plt.axis('off')
        plt.subplot(1,3,3)
        plt.imshow(mask_sample, cmap='gray')
        plt.title("Ground Truth")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("plots/{}.png".format(random.randint(0,99999)), dpi=500)
        plt.close()

# ============================
# Training Script
# ============================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script arguments")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device', type=int, default=0, help='Device id (e.g., 0 for cuda:0)')
    parser.add_argument('--subset_fraction', type=float, default=0.01, help='Fraction of the training dataset to use')
    parser.add_argument('--channels', type=str, default='rgb', choices=['rgb', 'all'], help='Use only RGB channels or all 7 channels')
    parser.add_argument('--model', type=str, default='unet',
                        choices=['unet','unetpp','deeplabv3+','fpn','pspnet'],
                        help='Which architecture to train')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    print(f"Seed set to: {args.seed}")
    print(f"Subset fraction: {args.subset_fraction}")
    print(f"Using channels: {args.channels}")

    # Paths to HDF5 files for training, validation, and testing.
    train_hdf5 = '/home/arra4944/projects/sustainbench/id_augmented_train_split_with_osm_new.h5'
    val_hdf5   = '/home/arra4944/projects/sustainbench/id_augmented_test_split_with_osm_new.h5'
    test_hdf5  = '/home/arra4944/projects/sustainbench/id_augmented_test_split_with_osm_new.h5'

    num_epochs = 20
    batch_size = 48
    learning_rate = 1e-4
    num_workers = 36

    print("Latest Training Script")

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform = None

    # Create training and validation datasets (subset only the training set).
    train_dataset = HDF5Dataset(train_hdf5, split='train', transform=transform, channels=args.channels)
    val_dataset   = HDF5Dataset(val_hdf5, split='test', transform=transform, channels=args.channels)

    num_samples = int(len(train_dataset) * args.subset_fraction)
    indices = torch.randperm(len(train_dataset))[:num_samples]
    train_dataset = Subset(train_dataset, indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, drop_last=False)

    # Initialize model based on channel selection.
    import segmentation_models_pytorch as smp
    if args.channels == "rgb":
        in_ch = 3
        if args.model == 'unet':
            model = smp.Unet(encoder_name="resnet34", in_channels=in_ch, classes=1)
        elif args.model == 'unetpp':
            model = smp.UnetPlusPlus(encoder_name="resnet34", in_channels=in_ch, classes=1)
        elif args.model == 'deeplabv3+':
            model = smp.DeepLabV3Plus(encoder_name="resnet50", in_channels=in_ch, classes=1)
        elif args.model == 'fpn':
            model = smp.FPN(encoder_name="resnet34", in_channels=in_ch, classes=1)
        elif args.model == 'pspnet':
            model = smp.PSPNet(encoder_name="resnet50", in_channels=in_ch, classes=1)
        model = model.to(device)
        # model = UNet(in_channels=3, out_channels=1).to(device)
    else:
        in_ch = 7
        if args.model == 'unet':
            model = smp.Unet(encoder_name="resnet34", in_channels=in_ch, classes=1)
        elif args.model == 'unetpp':
            model = smp.UnetPlusPlus(encoder_name="resnet34", in_channels=in_ch, classes=1)
        elif args.model == 'deeplabv3+':
            model = smp.DeepLabV3Plus(encoder_name="resnet50", in_channels=in_ch, classes=1)
        elif args.model == 'fpn':
            model = smp.FPN(encoder_name="resnet34", in_channels=in_ch, classes=1)
        elif args.model == 'pspnet':
            model = smp.PSPNet(encoder_name="resnet50", in_channels=in_ch, classes=1)
        model = model.to(device)
        # model = UNet(in_channels=7, out_channels=1).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.5, patience=5,
                                                           verbose=True)

    best_val_loss = float('inf')
    wandb.init(project="sustainbench-farm-id-newosm-ablations-all")
    wandb.log({'subset_size': args.subset_fraction, 'seed': args.seed, 'channels': args.channels, 'model': args.model})

    # Training + validation loop.
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss, train_dice, train_iou = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f} | Dice: {train_dice:.4f} | IoU: {train_iou:.4f}")
        
        val_loss, val_dice, val_iou = validate_one_epoch(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f} | Dice: {val_dice:.4f} | IoU: {val_iou:.4f}")
        
        scheduler.step(val_loss)
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     torch.save(model.state_dict(), 'best_unet_segmentation.pth')
        #     print("Best model saved.")
        wandb.log({'train_dice': train_dice, 'train_iou': train_iou,
                    'best_val_loss': best_val_loss, 'val_dice': val_dice,
                    'val_iou': val_iou, 'epoch': epoch + 1})
    
    print("\nTraining complete.")

    # After training, load the best model and evaluate on the test set.
    best_model = UNet(in_channels=3 if args.channels == "rgb" else 7, out_channels=1).to(device)
    best_model.load_state_dict(torch.load('best_unet_segmentation.pth'))
    best_model.eval()
    print("Evaluating on test set...")
    test_dataset = HDF5Dataset(test_hdf5, split='test', transform=transform, channels=args.channels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True, drop_last=False)
    test_loss, test_dice, test_iou = validate_one_epoch(best_model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Dice: {test_dice:.4f} | Test IoU: {test_iou:.4f}")

    # visualize_predictions(best_model, val_loader, device, channels_mode=args.channels)

    # Close HDF5 files.
    train_dataset.dataset.close()
    val_dataset.dataset.close()
    test_dataset.dataset.close()