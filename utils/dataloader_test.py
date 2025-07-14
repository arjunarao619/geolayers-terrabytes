import argparse
import json
import os
import random
from pathlib import Path

import geodatasets
import geopandas as gpd
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from geopandas import GeoDataFrame
from shapely.geometry import Point

from datasets.mmearth_dataset import get_mmearth_dataloaders
from geolayer_modalities.MMEarth_MODALITIES import (  # this contains all the input and output bands u need for pretraining.
    BASELINE_MODALITIES_IN,
    BASELINE_MODALITIES_OUT,
    MODALITIES_FULL,
)


def get_location(
    idx, phi_sin, phi_cos
):  # Convert sinusoidal encodings to lat/lon space for plotting
    sin_lat, cos_lat = phi_sin
    sin_lon, cos_lon = phi_cos

    lat = np.degrees(np.arctan2(cos_lat, sin_lat))
    lon = np.degrees(np.arctan2(cos_lon, sin_lon))

    return Point(lon, lat)


def plot_rgb_layer(train_dataloader):
    # Writing a naive plotting util to plot RGB imagery with their lat and lon as the title
    os.makedirs("/home/arjun/projects/geolayers/eda_plots", exist_ok=True)
    img = train_dataloader.__getitem__(random.randint(0, train_dataloader.__len__()))

    # To plot an RGB this function required a Sentinel-2 Modality in the input data
    assert "sentinel2" in img.keys()

    rgb = img["sentinel2"][(2, 1, 0), :, :].T * 255  # Grab RGB bands
    rgb = rgb.astype(np.uint8)
    imageio.imwrite(
        os.path.join("/home/arjun/projects/geolayers/eda_plots", img["id"] + ".png"),
        rgb,
    )


def plot_coverage_map():
    fig, ax = plt.subplots(figsize=(10, 6))
    df = pd.DataFrame()
    with open(
        "/home/arjun/datasets/data_100k_v001/data_100k_v001_tile_info.json", "r"
    ) as f:
        a = json.load(f)

    new_dict = {}
    # Fix strange bug where key in tile_info has one extra digit in ID compared to split
    for old_key in a.keys():
        if int(old_key) > 100000:
            new_key = str(int(old_key) // 10)
            new_dict[new_key] = a[old_key]
        else:
            new_dict[old_key] = a[old_key]

    del a

    with open(
        "/home/arjun/datasets/data_100k_v001/data_100k_v001_splits.json", "r"
    ) as f:
        alldata = json.load(f)

    train = alldata["train"]
    val = alldata["val"]

    world = gpd.read_file(geodatasets.data.naturalearth.land["url"])
    world.plot(ax=ax, color="white", edgecolor="black")

    for splitname, keys in [("train", train), ("val", val)]:
        lats = []
        lons = []
        for key in keys:
            try:
                lat = new_dict[str(key)]["lat"]
                lon = new_dict[str(key)]["lon"]
                lats.append(lat)
                lons.append(lon)
            except KeyError:
                continue

        # Convert lists to numpy arrays
        lats = np.array(lats)
        lons = np.array(lons)
        data = {"latitude": lats, "longitude": lons}

        df = pd.DataFrame(data)
        df["geometry"] = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]

        gdf = gpd.GeoDataFrame(df, geometry="geometry")
        gdf.set_crs(epsg=4326, inplace=True)
        mks = 0.009
        if splitname == "val":
            mks = 0.9
        gdf.plot(ax=ax, marker=".", label=splitname, alpha=0.7, markersize=mks)
        plt.legend()

    plt.title("MMEarth 100K pre-training dataset coverage on World Map")
    plt.savefig("/home/arjun/projects/geolayers/eda_plots/map1.png", dpi=200)


parser = argparse.ArgumentParser()
args = parser.parse_args()

# these 4 arguments need to be set manually
args.data_path = Path("/home/arjun/datasets/data_100k_v001")  # path to h5 file
args.random_crop = True  # ensure that if the dataset image size is 128 x 128, the resulting image after cropping is 112 x 112.
args.random_crop_size = 112  # the size of the crop
args.batch_size = 1

# define the input and output bands for the dataset
args.inp_modalities = BASELINE_MODALITIES_IN
args.out_modalities = BASELINE_MODALITIES_OUT

args.modalities = args.inp_modalities.copy()
args.modalities.update(
    args.out_modalities
)  # args modalities is a dictionary of all the input and output bands.
args.modalities_full = (
    MODALITIES_FULL  # this is a dictionary of all the bands in the dataset.
)

args.no_ffcv = (
    False  # this flag allows you to load the ffcv dataloader or the h5 dataset.
)
args.processed_dir = None  # default is automatically created in the data path. this is the dir where the beton file for ffcv is stored
args.num_workers = 4  # number of workers for the dataloader
args.distributed = False  # if you are using distributed training, set this to True


def collate_fn(batch):  # only for non ffcv dataloader
    return_batch = {}
    ids = [b["id"] for b in batch]
    return_batch = {
        modality: torch.stack([b[modality] for b in batch], dim=0)
        for modality in args.modalities.keys()
    }
    return ids, return_batch


# train_dataloader = get_mmearth_dataloaders(
#     args.data_path,
#     args.processed_dir,
#     args.modalities,
#     splits = ["train"],
#     num_workers=args.num_workers,
#     batch_size_per_device=args.batch_size,
#     distributed=args.distributed,
# )[0]

plot_coverage_map()

# print(train_dataloader.keys())
