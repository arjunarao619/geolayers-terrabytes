# train_baseline.py

import argparse
import copy
import os
import pathlib
import random
import warnings
from collections import defaultdict
from pathlib import Path

import geobench
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchgeo
import torchgeo.samplers
import torchvision
import torchvision.transforms as T
import yaml
from torch.utils.data import Subset
from torcheval.metrics import (
    MulticlassAccuracy,
    MulticlassAUPRC,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MultilabelAccuracy,
    MultilabelAUPRC,
    MultilabelPrecisionRecallCurve,
    TopKMultilabelAccuracy,
)
from torchgeo.datasets import stack_samples
from torchgeo.datasets.utils import BoundingBox
from torchgeo.samplers import GridGeoSampler, RandomBatchGeoSampler, RandomGeoSampler
from torchmetrics import Accuracy
from tqdm import tqdm

import wandb
from geolayer_modalities.BigEarthNet_MODALITIES import *
from geolayer_modalities.Chesapeake_MODALITIES import *
from geolayer_modalities.EnviroAtlas_MODALITIES import *
from geolayer_modalities.EuroSAT_MODALITIES import *
from geolayer_modalities.MMEarth_MODALITIES import *
from geolayer_modalities.So2Sat_MODALITIES import *
from models.fcn import FCN
from models.ModelFactory import ModelFactory  # Import ModelFactory
from models.resnet import ResNet50, ResNet101
from models.unet import UNet
from utils.eurosat_transforms import MinMaxNormalize
from utils.ipm_transforms import center_crop, nodata_check, pad_to, preprocess_image

warnings.filterwarnings("ignore")

import hydra
from omegaconf import DictConfig

# Import the new get_datasets function
from datasets.GeoLayerDataset import get_datasets

"""
Standard Config File Training. Changed to Hydra 10/24
"""


# Override config file with hydra
@hydra.main(config_path="config", config_name="config_baseline_eurobuildings")
def train(cfg: DictConfig):
    print(f"Training with config: {cfg}")

    # Check GPU
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        raise ValueError("Please use GPUs to train these models.")

    # Set Seed
    seed = cfg["seed"]
    random.seed(seed)
    torch.manual_seed(seed)  # For torch-based operations if needed

    os.environ["GEO_BENCH_DIR"] = "/share/geobench"

    # Set hyperparameters
    EPOCH = int(cfg["epoch"])
    pre_epoch = int(cfg["pre_epoch"])
    BATCH_SIZE = int(cfg["batch_size"])
    LR = float(cfg["lr"])

    # Load datasets using the new function
    datasets = get_datasets(cfg)
    trainset = datasets["trainset"]
    valset = datasets["valset"]
    testset = datasets["testset"]
    CLASS_LABELS = datasets["CLASS_LABELS"]
    samplers = datasets.get("samplers", {})

    # Initialize samplers if they exist
    if cfg["train_dataset"] in ["chesapeake", "enviroatlas"]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))

        trainsampler = samplers.get("train", None)
        valsampler = samplers.get("val", None)
        testsampler = samplers.get("test", None)

    # Initialize dataloaders
    if cfg["train_dataset"] in ["chesapeake", "enviroatlas"]:
        # Define generator for reproducibility
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))

        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=BATCH_SIZE,
            num_workers=cfg["num_workers"],
            drop_last=False,
            sampler=trainsampler,
            generator=generator,
        )
        valloader = torch.utils.data.DataLoader(
            valset,
            sampler=valsampler,
            batch_size=BATCH_SIZE,
            num_workers=cfg["num_workers"],
            drop_last=False,
            generator=generator,
            collate_fn=stack_samples,
        )
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=28,  # Adjust as needed
            num_workers=cfg["num_workers"],
            sampler=testsampler,
            generator=generator,
        )

        # Lazy debug workaround (if needed)
        for sample in trainloader:
            inp_modalities = [0] * sample["image"].shape[1]
            break
    else:
        # Initialize dataloaders normally for other datasets
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=cfg["num_workers"],
            drop_last=False,
        )
        valloader = torch.utils.data.DataLoader(
            valset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg["num_workers"],
            drop_last=False,
        )
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=len(testset),  # For full test set evaluation
            shuffle=False,
            num_workers=cfg["num_workers"],
        )

    # Initialize Model using ModelFactory
    input_modality = eval(cfg["input_modality"])  # Safer alternative to eval
    num_input_channels = len(input_modality)
    model_name = cfg["modelname"]
    pretrained = bool(cfg["pretrained"])
    num_classes = cfg["num_classes"]

    # Instantiate ModelFactory and get the model
    model_factory = ModelFactory(
        model_name=model_name,
        input_channels=num_input_channels,
        num_classes=num_classes,
        pretrained=pretrained,
    )
    net = model_factory.get_model().to(device)

    # Start a new wandb run to track this script
    exp = wandb.init(
        project=cfg["project"],
        notes=cfg["experiment_name"],
        tags=cfg["tags"],
        config=dict(cfg),
    )

    # Define loss criterion
    if cfg["train_dataset"] in ["bigearthnet", "m-bigearthnet"]:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = optim.Adam(
        net.parameters(), lr=LR, weight_decay=float(cfg["weight_decay"])
    )
    # Uncomment if using SGD
    # optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=float(cfg['weight_decay']))

    # Learning Rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    MIN_LR = 5e-7
    smax = torch.nn.Softmax(dim=1).to(device)

    """
    Evaluation Metrics:
    We define torcheval evaluation metrics for three broad classification types:
    Multi-Class classification, Multi-Label classification, Semantic Segmentation
    """
    if cfg["train_dataset"] in ["bigearthnet", "m-bigearthnet"]:
        multilabelaccuracy = MultilabelAccuracy()
        topkmultilabelaccuracy = TopKMultilabelAccuracy(k=5)
        multilabelauprc = MultilabelAUPRC(num_labels=cfg["num_classes"])
        multilabelprecisionmecallcurve = MultilabelPrecisionRecallCurve(
            num_labels=cfg["num_classes"]
        )

        multilabelaccuracy_val = MultilabelAccuracy()
        topkmultilabelaccuracy_val = TopKMultilabelAccuracy(k=5)
        multilabelauprc_val = MultilabelAUPRC(num_labels=cfg["num_classes"])
        multilabelprecisionmecallcurve_val = MultilabelPrecisionRecallCurve(
            num_labels=cfg["num_classes"]
        )

        multilabelaccuracy_test = MultilabelAccuracy()
        topkmultilabelaccuracy_test = TopKMultilabelAccuracy(k=5)
        multilabelauprc_test = MultilabelAUPRC(num_labels=cfg["num_classes"])
    elif cfg["train_dataset"] in ["m-so2sat", "eurosat", "eurobuildings"]:
        multiclassaccuracy = MulticlassAccuracy(device=device)
        multiclassauprc = MulticlassAUPRC(num_classes=cfg["num_classes"], device=device)
        multiclassprecision = MulticlassPrecision(
            num_classes=cfg["num_classes"], device=device, average="weighted"
        )
        multiclassrecall = MulticlassRecall(
            num_classes=cfg["num_classes"], device=device, average="weighted"
        )
        multiclassf1score = MulticlassF1Score(
            num_classes=cfg["num_classes"], device=device, average="weighted"
        )

        multiclassaccuracy_val = MulticlassAccuracy(device=device)
        multiclassauprc_val = MulticlassAUPRC(
            num_classes=cfg["num_classes"], device=device
        )
        multiclassprecision_val = MulticlassPrecision(
            num_classes=cfg["num_classes"], device=device, average="weighted"
        )
        multiclassrecall_val = MulticlassRecall(
            num_classes=cfg["num_classes"], device=device, average="weighted"
        )
        multiclassf1score_val = MulticlassF1Score(
            num_classes=cfg["num_classes"], device=device, average="weighted"
        )

        multiclassaccuracy_test = MulticlassAccuracy(device=device)
        multiclassauprc_test = MulticlassAUPRC(
            num_classes=cfg["num_classes"], device=device
        )

        # multiclassconfusionmatrix = MulticlassConfusionMatrix(cfg['num_classes'])
        multiclassconfusionmatrix_val = MulticlassConfusionMatrix(
            cfg["num_classes"], device=device
        )
    elif cfg["train_dataset"] in ["chesapeake", "enviroatlas"]:
        multiclassaccuracy = Accuracy(
            num_classes=cfg["num_classes"], task="multiclass", ignore_index=5
        ).to(device)
        multiclassaccuracy_val = Accuracy(
            num_classes=cfg["num_classes"], task="multiclass", ignore_index=5
        ).to(device)
        multiclassaccuracy_test = Accuracy(
            num_classes=cfg["num_classes"] + 1, task="multiclass", ignore_index=5
        ).to(device)

    # Initialize best validation variables
    best_val_epoch = 0
    best_val_weights = None
    # best_val_acc = 0  # Uncomment if tracking best validation accuracy

    # Main training loop
    if bool(cfg["dryrun"]):
        EPOCH = 1

    # Colormap normalization
    norm = mcolors.Normalize(vmin=0, vmax=11)  # Adjust if needed
    cmp = plt.get_cmap("tab20")

    for epoch in range(pre_epoch, EPOCH):
        print("\nEpoch: %d" % (epoch + 1))
        net.train()
        sum_loss = 0.0
        total = 0.0
        correct = 0.0

        print(scheduler.get_last_lr())
        for i, data in enumerate(trainloader, 0):
            # Prepare dataset
            if cfg["train_dataset"] == "bigearthnet":
                inputs, labels = (
                    data["image"][:, tuple(input_modality), ...],
                    data["label"],
                )
            elif cfg["train_dataset"] in ["m-bigearthnet", "m-so2sat"]:
                inputs, labels = data[0][:, tuple(input_modality), ...], data[1]
            elif cfg["train_dataset"] == "eurosat":
                inputs, labels = data["image"], data["label"]
            elif cfg["train_dataset"] == "eurobuildings":
                inputs, labels, mask = data["image"], data["label"], data["mask"]
                inputs = torch.cat(
                    (inputs, mask), dim=1
                )  # Input Stacking Geolayer Fusion
            elif cfg["train_dataset"] in ["chesapeake", "enviroatlas"]:
                inputs, labels = data["image"], data["mask"]
            else:
                raise ValueError(f"Unsupported train_dataset: {cfg['train_dataset']}")

            inputs, labels = inputs.to(device), labels.to(device)
            if epoch == EPOCH-1:
                from IPython import embed;embed()
            inputs = inputs.type(torch.float32)
            labels = labels.type(torch.float32)
            optimizer.zero_grad()

            # Forward pass
            outputs = net(inputs)

            # Compute loss
            if cfg["train_dataset"] in ["bigearthnet", "m-bigearthnet"]:
                loss = criterion(outputs, labels.float())
            elif cfg["train_dataset"] in ["chesapeake", "enviroatlas"]:
                loss = criterion(outputs, labels.squeeze(1).long())
            elif cfg["train_dataset"] in ["eurobuildings", "eurosat"]:
                loss = criterion(outputs, labels.long())
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Accumulate loss
            sum_loss += loss.item()

            # Compute accuracy and log metrics
            if cfg["classtype"] == "classification":
                probabilities = smax(outputs)
                predictions = torch.where(probabilities > 0.5, 1, 0)
                labels = labels.to(device, dtype=torch.int64)

                if cfg["train_dataset"] in ["bigearthnet", "m-bigearthnet"]:
                    total += labels.size(0) * cfg["num_classes"]
                    correct += predictions.eq(labels.data).cpu().sum()
                    overallaccuracy = (correct / total) * 100.0
                    wandb.log({"Train Prediction Overall Accuracy": overallaccuracy})
                    multilabelaccuracy.update(predictions, labels)
                    topkmultilabelaccuracy.update(probabilities, labels)
                    multilabelauprc.update(probabilities, labels)
                    wandb.log(
                        {"Train Multilabel Accuracy": multilabelaccuracy.compute()}
                    )
                    wandb.log(
                        {
                            "Train TopKMultilabelAccuracy Accuracy": topkmultilabelaccuracy.compute()
                        }
                    )
                    wandb.log({"Train AUPRC Accuracy": multilabelauprc.compute()})
                if cfg["train_dataset"] in ["m-so2sat", "eurosat", "eurobuildings"]:
                    class_predictions = torch.argmax(probabilities, dim=1).to(device)
                    multiclassaccuracy.update(class_predictions, labels)
                    multiclassprecision.update(class_predictions, labels)
                    multiclassrecall.update(class_predictions, labels)
                    multiclassf1score.update(class_predictions, labels)
                    multiclassauprc.update(probabilities, labels)

                    wandb.log(
                        {"Train MultiClass Accuracy": multiclassaccuracy.compute()}
                    )
                    wandb.log(
                        {
                            "Train MulticlassPrecision Accuracy": multiclassprecision.compute()
                        }
                    )
                    wandb.log(
                        {"Train Multiclassrecall Accuracy": multiclassrecall.compute()}
                    )
                    wandb.log(
                        {"Train Multiclassf1 Accuracy": multiclassf1score.compute()}
                    )
                    wandb.log(
                        {"Train Multiclassauprc Accuracy": multiclassauprc.compute()}
                    )
            elif cfg["classtype"] == "segmentation":
                preds = torch.argmax(outputs, dim=1)
                multiclassaccuracy.update(preds.flatten(), labels.flatten())
                wandb.log({"Train MultiClass Accuracy": multiclassaccuracy.compute()})

            # Log loss
            wandb.log(
                {"Loss": sum_loss / (i + 1), "epoch": epoch + (i / len(trainloader))}
            )
            print(
                "[epoch:%d, iter:%d] Loss: %.05f "
                % (epoch + 1, (i + 1 + epoch * len(trainloader)), sum_loss / (i + 1))
            )

        # Step the scheduler
        scheduler.step()

        # Enforce minimum learning rate
        for param_group in optimizer.param_groups:
            if param_group["lr"] < MIN_LR:
                param_group["lr"] = MIN_LR

        # Optional: Validation Loop (Currently Commented Out)
        """
        total_val = 0
        correct_val = 0
        with torch.no_grad():
            net.eval()
            for i, data in enumerate(valloader, 0):
                if cfg['train_dataset'] == 'bigearthnet':
                    inputs, label = data['image'][:, tuple(input_modality), ...], data['label']
                elif cfg['train_dataset'] in ['m-bigearthnet', 'm-so2sat']:
                    inputs, label = data[0][:, tuple(input_modality), ...], data[1]
                elif cfg['train_dataset'] == 'eurosat':
                    inputs, label = data['image'], data['label']
                elif cfg['train_dataset'] == 'eurobuildings':
                    inputs, label, mask = data['image'], data['label'], data['mask']
                    inputs = torch.cat((inputs, mask), dim=1)
                elif cfg['train_dataset'] in ['chesapeake', 'enviroatlas']:
                    inputs, label = data['image'], data['mask']
                else:
                    raise ValueError(f"Unsupported train_dataset: {cfg['train_dataset']}")

                inputs = inputs.to(device).to(torch.float32)
                label = label.to(device)
                raw_out = net(inputs)

                if cfg['train_dataset'] in ['m-so2sat', 'eurosat', 'eurobuildings']:
                    _, predicted = torch.max(raw_out.data, 1)
                    total_val += label.size(0)
                    correct_val += predicted.eq(label.data).cpu().sum()
                    wandb.log({"Prediction Accuracy Val": (correct_val / total_val) * 100.0})
                elif cfg['train_dataset'] in ['chesapeake', 'enviroatlas']:
                    preds = torch.argmax(raw_out, dim=1)
                    multiclassaccuracy_val.update(preds.flatten(), label.flatten())

            current_val_acc = multiclassaccuracy_val.compute()
            if current_val_acc > best_val_acc:
                best_val_model = copy.deepcopy(net)
                best_val_acc = current_val_acc
                best_val_epoch = epoch
        """

    # Testing Loop
    total = 0
    correct = 0
    test_tiles = 0
    for i, data in tqdm(enumerate(testloader, 0), total=len(testloader)):
        with torch.no_grad():
            net.eval()
            if cfg["train_dataset"] == "bigearthnet":
                inputs, label = (
                    data["image"][:, tuple(input_modality), ...],
                    data["label"],
                )
            elif cfg["train_dataset"] in ["m-bigearthnet", "m-so2sat"]:
                inputs, label = data[0][:, tuple(input_modality), ...], data[1]
            elif cfg["train_dataset"] == "eurosat":
                inputs, label = data["image"], data["label"]
            elif cfg["train_dataset"] == "eurobuildings":
                inputs, label, mask = data["image"], data["label"], data["mask"]
                inputs = torch.cat((inputs, mask), dim=1)
            elif cfg["train_dataset"] in ["chesapeake", "enviroatlas"]:
                inputs, label = data["image"], data["mask"]
                test_tiles += inputs.shape[0]
            else:
                raise ValueError(f"Unsupported train_dataset: {cfg['train_dataset']}")

            inputs = inputs.to(device).to(torch.float32)
            label = label.to(device)
            raw_out = net(inputs)

            if cfg["train_dataset"] in ["m-so2sat", "eurosat", "eurobuildings"]:
                _, predicted = torch.max(raw_out.data, 1)
                total += label.size(0)
                correct += predicted.eq(label.data).cpu().sum()
                multiclassaccuracy_test.update(predicted, label)
                wandb.log({"Prediction Accuracy Test": (correct / total) * 100.0})
            elif cfg["train_dataset"] in ["chesapeake", "enviroatlas"]:
                preds = torch.argmax(raw_out, dim=1)
                multiclassaccuracy_test.update(preds.flatten(), label.flatten())

    if cfg["train_dataset"] in ["chesapeake", "enviroatlas"]:
        wandb.log({"Test MultiClass Accuracy": multiclassaccuracy_test.compute()})
        wandb.log({"Total Test Tile Count": test_tiles})
    elif cfg["train_dataset"] in ["m-so2sat", "eurosat", "eurobuildings"]:
        wandb.log({"Test Prediction Accuracy": (correct / total) * 100.0})
        wandb.log({"TorchEval Prediction Accuracy": multiclassaccuracy_test.compute()})

    if cfg["train_dataset"] in ["chesapeake", "enviroatlas"]:
        print(multiclassaccuracy_test.compute())
    else:
        print(f"Test Accuracy: {(correct / total) * 100.0:.2f}%")

    # Optionally log best validation metrics
    # wandb.log({"Best Val Epoch": best_val_epoch})
    # wandb.log({"Best Val Accuracy": best_val_acc})

    print("Training has finished, total epochs: %d" % EPOCH)


# Used to calculate subsets in ChesapeakeCVPR datasets (Can be extended to any torchgeo RasterDataset)
def calculate_aoi_bounds(subset_size, original_bounds) -> BoundingBox:
    """
    Calculate the AOI bounds for a given subset size.

    Parameters:
    - subset_size (float): Fraction of the area to use (e.g., 0.01 for 1%).
    - original_bounds (list): The bounding box [minx, maxx, miny, maxy].

    Returns:
    - aoi_bounds (tuple): The AOI bounds in EPSG:3857.
    """
    minx, maxx, miny, maxy, mint, maxt = original_bounds

    # Calculate width and height of the original bounding box
    width = maxx - minx
    height = maxy - miny

    # Calculate width and height of the AOI
    aoi_width = width * (subset_size**0.5)
    aoi_height = height * (subset_size**0.5)

    # Define AOI bounds, placing it in the lower-left corner of the original bound
    aoi_minx, aoi_miny = minx, miny
    aoi_maxx = minx + aoi_width
    aoi_maxy = miny + aoi_height

    aoi_trim = BoundingBox(
        minx=aoi_minx, maxx=aoi_maxx, miny=aoi_miny, maxy=aoi_maxy, mint=mint, maxt=maxt
    )

    return aoi_trim


if __name__ == "__main__":
    train()
