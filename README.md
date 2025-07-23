# Using Multiple Input Modalities Can Improve Data-Efficiency and O.O.D. Generalization for ML with Satellite Imagery

[![Hugging Face Dataset](https://img.shields.io/badge/HuggingFace-Dataset-orange?logo=huggingface)](https://huggingface.co/datasets/arjunrao2000/geolayers)

This repository contains training scripts to train commonly available deep-learning-based models on our augmented benchmark dataset release, as reported in our paper "Using Multiple Input Modalities Can Improve Data-Efficiency and O.O.D. Generalization for ML with Satellite Imagery." If you use our datasets or code in your research, please cite our work (BibTeX below).

## EnviroAtlas Multi-Modal Land Cover Segmentation


This repository implements multi-modal learning approaches for satellite imagery segmentation on the EnviroAtlas dataset, demonstrating how fusing geographic data layers with optical imagery improves model performance in data-efficient and out-of-distribution settings.

### Getting Started

#### Prerequisites
```bash
# Required packages
torch>=1.10.0
torchgeo==0.7.0
hydra-core
wandb
numpy
matplotlib
seaborn
```

### Architecture

#### Core Components

1. **Main Training Script**: `train_baseline.py`
   - Uses Hydra for configuration management
   - Implements semantic segmentation with FCN model

2. **Modality Definitions**: `geolayer_modalities/EnviroAtlas_MODALITIES.py`
   - Defines input channel combinations for different fusion strategies
   - Key modalities:
     - `NAIP_ONLY`: Channels 1-4 (RGB + NIR only)
     - `NAIP_P_PRIOR`: Channels 1-5 (RGB + NIR + prior layer)
     - Additional combinations with roads, water, buildings, etc.

3. **Experiment Scripts**: `experiment_bash_scripts/enviroatlas/ipm_label_efficiency.sh`
   - Automated label efficiency experiments
   - Multi-GPU parallel execution
   - Systematic evaluation across subset sizes and random seeds

   Note that we run our experiments on the EnviroAtlas dataset using 4 NVIDIA RTX 8000 GPUs with a total 36 CPU workers. Adjust the bash script to meet your compute availability.

#### Model Details
- **Architecture**: 5-layer Fully Convolutional Network (FCN)
- **Input Channels**: Variable based on modality (4-12 channels)
- **Output**: 5 land cover classes + 1 ignore class (index 5)
- **Loss**: CrossEntropyLoss
- **Optimizer**: Adam with StepLR scheduler


#### Configuration

The training script uses Hydra configuration. Key parameters:
- `device`: GPU device (e.g., "cuda:0")
- `input_modality`: Channel selection (e.g., "NAIP_ONLY", "NAIP_P_PRIOR")
- `test_cities`: List of test cities for evaluation
- `subset_size`: Fraction of training data to use (0.01 to 1.0)
- `seed`: Random seed for reproducibility
- `epoch`: Number of training epochs
- `batch_size`: Training batch size (default: 128)
- `lr`: Learning rate (default: 1e-3)

#### Running Single Experiments

Basic training with NAIP imagery only:
```bash
python train_baseline.py \
    ++input_modality=NAIP_ONLY \
    ++test_cities="['pittsburgh_pa-2010_1m']" \
    ++subset_size=1.0 \
    ++seed=42 \
    ++device=cuda:0
```

Training with prior layer (PROC-STACK fusion):
```bash
python train_baseline.py \
    ++input_modality=NAIP_P_PRIOR \
    ++test_cities="['austin_tx-2012_1m', 'durham_nc-2012_1m']" \
    ++subset_size=0.1 \
    ++seed=42 \
    ++epoch=70 \
    ++device=cuda:0
```

#### Running Label Efficiency Experiments

The provided bash script runs comprehensive label efficiency experiments:

```bash
bash experiment_bash_scripts/enviroatlas/ipm_label_efficiency.sh
```

This script:
- Tests 9 subset sizes: [1.0, 0.75, 0.5, 0.35, 0.2, 0.1, 0.05, 0.02, 0.01]
- Uses 5 random seeds: [52, 62, 72, 82, 92]
- Evaluates 2 modalities: NAIP_ONLY, NAIP_P_PRIOR
- Tests on 3 cities:
  - `pittsburgh_pa-2010_1m` (training city)
  - `austin_tx-2012_1m` (OOD test)
  - `durham_nc-2012_1m` (OOD test)
- Automatically adjusts epochs based on subset size (base_epochs / subset_size)
- Runs 4 experiments in parallel across GPUs

#### Available Input Modalities

From `EnviroAtlas_MODALITIES.py`:

| Modality | Channels | Description |
|----------|----------|-------------|
| `NAIP_ONLY` | 1-4 | RGB + NIR baseline |
| `NAIP_P_PRIOR` | 1-5 | RGB + NIR + hand-crafted prior |
| `NAIP_P_ROADS` | 1-5 | RGB + NIR + roads layer |
| `NAIP_P_WATER` | 1-5 | RGB + NIR + water layer |
| `NAIP_P_BUILDINGS` | 1-5 | RGB + NIR + buildings layer |
| `NAIP_P_WATER_P_WATERBODIES` | 1-6 | RGB + NIR + water + waterbodies |
| `ALL_ENVIRO` | 1-12 | All available layers |

#### Data Organization

The script expects EnviroAtlas data organized for TorchGeo compatibility:
- NAIP imagery (4 channels: RGB + NIR)
- Land cover masks (5 classes + nodata)
- Geographic layers (roads, water, buildings, prior)
- Cities: Pittsburgh (train), Austin & Durham (test)

### Key Implementation Details

#### Data Loading
- Uses TorchGeo's `RandomGeoSampler` for training (128x128 patches)
- Uses `GridGeoSampler` for validation/testing (256x256 patches)
- Implements `stack_samples` collate function for geographic data

## SustainBench Field Boundary Delineation
The `sustainbench/` directory trains segmentation models for agricultural field boundary delineation using the SustainBench dataset, demonstrating how incorporating OpenStreetMap (OSM) data and Digital Elevation Models (DEM) with RGB satellite imagery improves segmentation performance across different data efficiency regimes.

### 🚀 Getting Started
#### Prerequisites
```bash#
Required packages
torch>=1.10.0
segmentation_models_pytorch
h5py
hydra-core
wandb
numpy
matplotlib
tqdm
```

### File structure

Main Training Script: `sustainbench/train_unet.py`
* Implements multiple segmentation architectures via segmentation_models_pytorch
* Supports various input channel combinations
* Binary segmentation for field boundary detection


Composite Model Script: `sustainbench/train_unet_proc_stack.py`

* Implements FCN + UNet architecture for processing auxiliary channels
* FCN processes geographic layers before fusion with RGB
* Experimental approach to study arbitrary fusion mechanisms.


### Experiment Scripts:

* `sustainbench/experiments.sh`: Baseline RGB-only experiments
* `sustainbench/ablations_isolate.sh`: Channel ablation studies (OSM vs DEM)
* `sustainbench/ablations_model.sh` : Architecture comparison experiments (Reproduce Figure 7 in paper)

Note: Experiments are configured for 4 NVIDIA GPUs with 36 CPU workers. Adjust the bash scripts to match your compute availability.

### Model Details

* Architectures: UNet (ResNet34), UNet++, DeepLabV3+ (ResNet50), FPN, PSPNet
* Input Channels: Variable based on modality (3-7 channels)
* Output: Binary mask (field boundaries)
* Loss: BCEWithLogitsLoss
* Metrics: Dice coefficient, IoU

### Configuration
Key parameters for training scripts:

* --model: Architecture choice (default: "unet")
* --channels: Input modality selection ("rgb", "osm", "dem", "all")
* --subset_fraction: Fraction of training data (0.01 to 1.0)
* --seed: Random seed for reproducibility
* --device: GPU device ID
* --fcn_out_channels: Output channels for FCN in composite model (`train_unet_proc_stack.py` only)

#### Running Single Experiments
Basic training with RGB imagery only:
```bash
python sustainbench/train_unet.py \
    --model unet \
    --channels rgb \
    --subset_fraction 1.0 \
    --seed 42 \
    --device 0
```
Training with RGB + OSM layers:
```bash
python sustainbench/train_unet.py \
    --model unet \
    --channels osm \
    --subset_fraction 0.1 \
    --seed 42 \
    --device 0
```
Training with composite model (FCN preprocessing):
```bash
python sustainbench/train_unet_proc_stack.py \
    --channels all \
    --subset_fraction 0.1 \
    --seed 42 \
    --fcn_out_channels 3 \
    --device 0
```
Running Systematic Experiments
For comprehensive label efficiency experiments across modalities:
```bash
cd sustainbench
bash ablations_isolate.sh
```
For model architecture comparisons:
```bash
cd sustainbench
bash ablations_model.sh
```
These scripts evaluate:

* 9 subset sizes: [0.01, 0.05, 0.1, 0.15, 0.2, 0.35, 0.5, 0.75, 1.0]
* 5 random seeds: [42, 1, 1234, 0, 22]
* Multiple modalities: RGB, RGB+OSM, RGB+DEM, All channels
* 5 architectures: UNet, UNet++, DeepLabV3+, FPN, PSPNet
* Runs 4 experiments in parallel across GPUs

Available Input Modalities
| Modality | Channels | Description |
|----------|----------|-------------|
| rgb | 1-3 | RGB baseline |
| osm | 1-6 | RGB + OpenStreetMap layers (roads, buildings, water) |
| dem | 1-4 | RGB + Digital Elevation Model |
| all | 1-7 | RGB + OSM + DEM (all available layers) |



```
@inproceedings{
  rao2025using,
  title={Using Multiple Input Modalities can Improve Data-Efficiency and O.O.D. Generalization for {ML} with Satellite Imagery},
  author={Arjun Rao and Esther Rolf},
  booktitle={TerraBytes - ICML 2025 workshop},
  year={2025},
  url={https://www.arxiv.org/abs/2507.13385}
}
```