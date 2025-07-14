#!/bin/bash
MAX_JOBS=4
devices=(0 1 2 3)
# Only UNet
MODELS=(unet)
# Channel modes: RGB+OSM (6 channels) and RGB+DEM (4 channels)
CHANNELS=(osm dem)
SEEDS=(42 1 1234 0 22)
SUBSETS=(0.01 0.05 0.1 0.15 0.2 0.35 0.5 0.75 1.0)

run_cmd() {
    "$@" &
    while [ "$(jobs -r | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 1
    done
}

job_count=0
for model in "${MODELS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for subset in "${SUBSETS[@]}"; do
      for channel in "${CHANNELS[@]}"; do
        # round-robin device selection
        dev=${devices[$(( job_count % ${#devices[@]} ))]}
        echo "Running UNet | Seed: ${seed} | Subset: ${subset} | Channels: ${channel} (mode) on device ${dev}"
        run_cmd python train_unet.py \
                --model "$model" \
                --subset_fraction "$subset" \
                --seed "$seed" \
                --channels "$channel" \
                --device "$dev"
        job_count=$((job_count+1))
      done
    done
  done
done

wait
echo "All experiments completed."
