#!/bin/bash
MAX_JOBS=4
devices=(0 1 2 3)
MODELS=(unet unetpp deeplabv3+ fpn pspnet)
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
      # pick a device in round-robin
      dev=${devices[$(( job_count % ${#devices[@]} ))]}
      run_cmd python train_unet.py \
              --model "$model" \
              --subset_fraction "$subset" \
              --seed "$seed" \
              --channels all \
              --device "$dev"
      job_count=$((job_count+1))
    done
  done
done

wait
echo "All experiments completed."
