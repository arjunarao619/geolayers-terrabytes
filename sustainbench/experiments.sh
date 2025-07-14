#!/bin/bash
# run_4_at_a_time.sh
#
# This script runs your experiments while ensuring that no more than 4 experiments
# run concurrently. Once 4 jobs are running, the script waits before launching new ones.
#
# It cycles over your seeds: 0, 1, 1234, 42, 22 and runs the following experiments:
#
#   Without --use_osm:
#     python train_unet.py --subset_fraction 0.01 --seed <seed> --device 0
#     python train_unet.py --subset_fraction 0.05 --seed <seed> --device 1
#     python train_unet.py --subset_fraction 0.1  --seed <seed> --device 2
#     python train_unet.py --subset_fraction 0.15 --seed <seed> --device 2
#     python train_unet.py --subset_fraction 0.2  --seed <seed> --device 2
#     python train_unet.py --subset_fraction 0.35 --seed <seed> --device 3
#     python train_unet.py --subset_fraction 0.5  --seed <seed> --device 0
#     python train_unet.py --subset_fraction 0.75 --seed <seed> --device 1
#     python train_unet.py --subset_fraction 1    --seed <seed> --device 2
#
#   With --use_osm:
#     python train_unet.py --subset_fraction 0.01 --seed <seed> --use_osm --device 3
#     python train_unet.py --subset_fraction 0.05 --seed <seed> --use_osm --device 0
#     python train_unet.py --subset_fraction 0.1  --seed <seed> --use_osm --device 1
#     python train_unet.py --subset_fraction 0.15 --seed <seed> --use_osm --device 1
#     python train_unet.py --subset_fraction 0.2  --seed <seed> --use_osm --device 1
#     python train_unet.py --subset_fraction 0.35 --seed <seed> --use_osm --device 2
#     python train_unet.py --subset_fraction 0.5  --seed <seed> --use_osm --device 3
#     python train_unet.py --subset_fraction 0.75 --seed <seed> --use_osm --device 0
#     python train_unet.py --subset_fraction 1    --seed <seed> --use_osm --device 1
#
# Change the "python" command if you need to force Python 3 (e.g. "python3").

# Maximum number of concurrent jobs.
MAX_JOBS=4

# run_cmd runs a command in the background and then waits if thse
# number of running jobs is at least MAX_JOBS.
run_cmd() {
    "$@" &
    # Wait until the number of running background jobs is less than MAX_JOBS.
    while [ "$(jobs -r | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 1
    done
}

# Define the seeds to use.
for seed in 42 1 1234 0 22; do

    # Without --use_osm
    run_cmd python train_unet.py --subset_fraction 0.01 --seed "$seed" --device 0
    run_cmd python train_unet.py --subset_fraction 0.05 --seed "$seed" --device 1
    run_cmd python train_unet.py --subset_fraction 0.1  --seed "$seed" --device 2
    run_cmd python train_unet.py --subset_fraction 0.15 --seed "$seed" --device 3
    run_cmd python train_unet.py --subset_fraction 0.2  --seed "$seed" --device 0
    run_cmd python train_unet.py --subset_fraction 0.35 --seed "$seed" --device 1
    run_cmd python train_unet.py --subset_fraction 0.5  --seed "$seed" --device 2
    run_cmd python train_unet.py --subset_fraction 0.75 --seed "$seed" --device 3
    run_cmd python train_unet.py --subset_fraction 1    --seed "$seed" --device 0

    # # With --use_osm


    # run_cmd python train_unet_proc_stack.py --subset_fraction 0.01 --seed "$seed" --channels all --device 0 --fcn_out_channels 3
    # run_cmd python train_unet_proc_stack.py --subset_fraction 0.05 --seed "$seed" --channels all --device 1 --fcn_out_channels 3
    # run_cmd python train_unet_proc_stack.py --subset_fraction 0.1  --seed "$seed" --channels all --device 2 --fcn_out_channels 3
    # run_cmd python train_unet_proc_stack.py --subset_fraction 0.15 --seed "$seed" --channels all --device 3 --fcn_out_channels 3
    # run_cmd python train_unet_proc_stack.py --subset_fraction 0.2  --seed "$seed" --channels all --device 0 --fcn_out_channels 3
    # run_cmd python train_unet_proc_stack.py --subset_fraction 0.35 --seed "$seed" --channels all --device 1 --fcn_out_channels 3
    # run_cmd python train_unet_proc_stack.py --subset_fraction 0.5  --seed "$seed" --channels all --device 2 --fcn_out_channels 3
    # run_cmd python train_unet_proc_stack.py --subset_fraction 0.75 --seed "$seed" --channels all --device 3 --fcn_out_channels 3
    # run_cmd python train_unet_proc_stack.py --subset_fraction 1    --seed "$seed" --channels all --device 0 --fcn_out_channels 3

    # run_cmd python train_unet_proc_stack.py --subset_fraction 0.01 --seed "$seed" --channels all --device 0
    # run_cmd python train_unet_proc_stack.py --subset_fraction 0.05 --seed "$seed" --channels all --device 1
    # run_cmd python train_unet_proc_stack.py --subset_fraction 0.1  --seed "$seed" --channels all --device 2
    # run_cmd python train_unet_proc_stack.py --subset_fraction 0.15 --seed "$seed" --channels all --device 3
    # run_cmd python train_unet_proc_stack.py --subset_fraction 0.2  --seed "$seed" --channels all --device 0
    # run_cmd python train_unet_proc_stack.py --subset_fraction 0.35 --seed "$seed" --channels all --device 1
    # run_cmd python train_unet_proc_stack.py --subset_fraction 0.5  --seed "$seed" --channels all --device 2
    # run_cmd python train_unet_proc_stack.py --subset_fraction 0.75 --seed "$seed" --channels all --device 3
    # run_cmd python train_unet_proc_stack.py --subset_fraction 1    --seed "$seed" --channels all --device 0
done

# Wait for any remaining background jobs to complete.
wait
echo "All experiments completed."
