import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune ViT on CIFAR-10")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save model checkpoints")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    return parser.parse_args()

def get_dataloaders(batch_size, num_workers):
    from transformers import AutoFeatureExtractor

    # Use the feature extractor from the pretrained model
    feature_extractor = AutoFeatureExtractor.from_pretrained("mrm8488/vit-base-patch16-224-pretrained-cifar10")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize CIFAR-10 images to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
    ])
    
    # Specify the root directory for the dataset
    root_dir = "/scratch/local/arra4944_images"
    
    trainset = datasets.CIFAR10(root=root_dir, train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root=root_dir, train=False, download=True, transform=transform)
    
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, test_loader


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    accuracy = 100.0 * correct / total
    return total_loss / len(dataloader), accuracy

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    return total_loss / len(dataloader), accuracy

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained Hugging Face model
    model = AutoModelForImageClassification.from_pretrained("mrm8488/vit-base-patch16-224-pretrained-cifar10")
    model.classifier = torch.nn.Linear(model.config.hidden_size, 10)

    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Dataloaders
    train_loader, test_loader = get_dataloaders(args.batch_size, args.num_workers)

    # Training loop
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(f"Epoch [{epoch + 1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        # Save model checkpoint
        model.save_pretrained(os.path.join(args.output_dir, f"epoch_{epoch + 1}"))

if __name__ == "__main__":
    main()
