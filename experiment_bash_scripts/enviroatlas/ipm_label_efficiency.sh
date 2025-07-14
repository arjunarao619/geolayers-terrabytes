#!/bin/bash

# Subset sizes and random seeds
SUBSET_SIZES=(1.0 0.75 0.5 0.35 0.2 0.1 0.05 0.02 0.01)
SEEDS=(52 62 72 82 92)
MODALITIES=("NAIP_ONLY" "NAIP_P_PRIOR")
CITIES=("pittsburgh_pa-2010_1m" "austin_tx-2012_1m" "durham_nc-2012_1m")
NUM_CPU_CORES=$(nproc) # Total CPU cores
NUM_EXPERIMENTS=4       # Experiments running simultaneously
WORKERS=$((NUM_CPU_CORES / NUM_EXPERIMENTS))

# Function to calculate epochs (floating-point compatible)
calculate_epochs() {
    local base_epochs=7
    local subset=$1
    echo | awk -v base="$base_epochs" -v subset="$subset" '{printf "%.0f", base / subset}'
}

# Function to run experiments on 8 GPUs at a time
run_batch() {
    local SUBSET=$1
    local SEED=$2
    local MODALITY=$3
    local CITY=$4
    local GPU_ID=$5

    EPOCHS=$(calculate_epochs $SUBSET)
    echo "Running experiments for modality: $MODALITY, city: $CITY, subset size: $SUBSET, seed: $SEED with $WORKERS workers per DataLoader and $EPOCHS epochs"

    python train_baseline.py ++epoch=$EPOCHS ++device=cuda:$GPU_ID ++input_modality=$MODALITY \
        ++test_cities="['$CITY']" ++subset_size=$SUBSET ++seed=$SEED \
        ++num_workers=$WORKERS ++tags="['label-efficiency', '$MODALITY', 'reproduce IPM']" &
}

# Main loop to manage GPU assignments and batch processing
GPU_ID=0
for MODALITY in "${MODALITIES[@]}"; do
    for SUBSET in "${SUBSET_SIZES[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            for CITY in "${CITIES[@]}"; do
                run_batch $SUBSET $SEED $MODALITY $CITY $GPU_ID
                GPU_ID=$(( (GPU_ID + 1) % NUM_EXPERIMENTS ))
                if [ $GPU_ID -eq 0 ]; then
                    wait # Ensure all GPUs are synchronized after a full round
                fi
            done
        done
    done
done
wait # Catch any remaining processes

echo "All experiments completed."
