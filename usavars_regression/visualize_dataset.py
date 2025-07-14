#!/usr/bin/env python3
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter
import torchvision.transforms as T
from torchgeo.datasets import USAVars as usavars

# Import your OSM helper functions.
from utils.get_osm import compute_bbox_from_centroid, get_osm_raster_from_centroid, get_osm_raster
# Also import your WorldPop and DEM functions.
from utils.get_topo import get_population_raster, get_dem_raster

def normalize(img):
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    return img

def aggregate_osm(raster_array, sigma):
    """
    Given a 19-channel OSM raster (shape: H x W x 19),
    smooth each channel with a Gaussian filter (sigma),
    normalize each channel to [0,1], and aggregate them
    into a single 3-channel RGB composite using fixed color weighting.
    """
    H, W, C = raster_array.shape
    smoothed = np.empty_like(raster_array, dtype=np.float32)
    for i in range(C):
        smoothed[..., i] = gaussian_filter(raster_array[..., i].astype(np.float32), sigma=sigma)
    normalized = np.empty_like(smoothed)
    for i in range(C):
        channel = smoothed[..., i]
        if channel.max() > channel.min():
            normalized[..., i] = (channel - channel.min()) / (channel.max() - channel.min())
        else:
            normalized[..., i] = channel
    # Use 19 distinct colors from tab20.
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in range(19)]
    composite = np.zeros((H, W, 3), dtype=np.float32)
    for i, color in enumerate(colors):
        r, g, b, _ = color
        composite[..., 0] += normalized[..., i] * r
        composite[..., 1] += normalized[..., i] * g
        composite[..., 2] += normalized[..., i] * b
    composite = composite - composite.min()
    if composite.max() > 0:
        composite = composite / composite.max()
    return composite

def visualize_sample(sample, img_size=256, gsd=4.0, sigma=1.0):
    """
    For a given USAVars sample, compute its bounding box from the centroid and then
    retrieve and pre-process three rasters:
      - OSM composite: obtained by fetching a 19-channel OSM raster and aggregating it into an RGB image.
      - WorldPop: population density raster, converted to RGB by repeating the single band.
      - DEM: digital elevation model, colorized using the terrain colormap.
      
    Also, extract the base RGB image from the sample (using the first three channels).
    
    Returns:
      base_rgb, osm_composite, worldpop_rgb, dem_rgb
      Each image is of shape (img_size, img_size, 3).
    """
    # Base RGB image.
    base_img = sample['image'].numpy()  # (4, H, W)
    base_rgb = base_img[:3, ...].transpose(1, 2, 0)
    base_rgb = normalize(base_rgb)
    
    # Compute bounding box.
    centroid_lat = sample['centroid_lat'].item()
    centroid_lon = sample['centroid_lon'].item()
    bbox = compute_bbox_from_centroid(centroid_lat, centroid_lon, img_size, gsd)
    
    # OSM composite.
    osm_raster, _ = get_osm_raster_from_centroid(centroid_lat, centroid_lon, img_size, gsd)
    # osm_raster shape: (img_size, img_size, 19)
    osm_composite = aggregate_osm(osm_raster, sigma=sigma)  # (img_size, img_size, 3)
    
    # WorldPop raster.
    worldpop_raster, _ = get_population_raster(bbox, img_size, img_size)
    # Create an RGB image by repeating the band.
    worldpop_rgb = np.dstack([worldpop_raster]*3)
    worldpop_rgb = normalize(worldpop_rgb)
    
    # DEM raster.
    dem, _ = get_dem_raster(bbox, img_size, img_size)
    dem = np.asarray(dem, dtype=np.float32)
    # If dem is a masked array, fill missing values.
    if hasattr(dem, "filled"):
        dem = dem.filled(np.nan)
    cmap_dem = plt.get_cmap('terrain')
    dem_rgb = cmap_dem(dem)[:, :, :3]  # Convert to RGB (drop alpha)
    dem_rgb = normalize(dem_rgb)
    
    return base_rgb, osm_composite, worldpop_rgb, dem_rgb

def main():
    # Load a sample from USAVars training dataset.
    train_ds = usavars(
        root='/share/usavars',
        split='train',
        labels=('treecover',),
        download=False,
        checksum=False
    )
    total_samples = len(train_ds)
    print(f"Total samples: {total_samples}")
    # Select one random sample.
    sample_idx = random.randrange(total_samples)
    sample = train_ds.__getitem__(sample_idx)
    
    # Get visualizations.
    img_size = 256
    gsd = 4.0
    sigma = 1.0
    base_rgb, osm_img, worldpop_img, dem_img = visualize_sample(sample, img_size=img_size, gsd=gsd, sigma=sigma)
    
    # Plot in a 2x2 grid.
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes[0, 0].imshow(base_rgb)
    axes[0, 0].set_title("Base RGB")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(osm_img)
    axes[0, 1].set_title("OSM Composite")
    axes[0, 1].axis("off")
    
    axes[1, 0].imshow(worldpop_img)
    axes[1, 0].set_title("WorldPop Density")
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(dem_img)
    axes[1, 1].set_title("DEM (Elevation)")
    axes[1, 1].axis("off")
    
    plt.tight_layout()
    plt.savefig("sample_topo_visualization.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("Visualization saved as sample_topo_visualization.png")

if __name__ == "__main__":
    main()
