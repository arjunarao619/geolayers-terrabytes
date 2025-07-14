#!/bin/bash

# python train_baseline.py ++epoch=7 ++device=cuda:0 ++input_modality=NAIP_ONLY ++test_cities="['pittsburgh_pa-2010_1m']" ++subset_size=1 ++seed=42 ++tags="['label-efficiency', 'NAIP_ONLY', 'reproduce IPM']" &
# python train_baseline.py ++epoch=7 ++device=cuda:1 ++input_modality=NAIP_ONLY ++test_cities="['durham_nc-2012_1m']" ++subset_size=1 ++seed=42 ++tags="['label-efficiency', 'NAIP_ONLY', 'reproduce IPM']" &
# python train_baseline.py ++epoch=7 ++device=cuda:2 ++input_modality=NAIP_ONLY ++test_cities="['austin_tx-2012_1m']" ++subset_size=1 ++seed=42 ++tags="['label-efficiency', 'NAIP_ONLY', 'reproduce IPM']" &
# python train_baseline.py ++epoch=7 ++device=cuda:3 ++input_modality=NAIP_ONLY ++test_cities="['phoenix_az-2010_1m']" ++subset_size=1 ++seed=42 ++tags="['label-efficiency', 'NAIP_ONLY', 'reproduce IPM']" &


# python train_baseline.py ++epoch=7 ++device=cuda:0 ++input_modality=NAIP_P_PRIOR ++test_cities="['pittsburgh_pa-2010_1m']" ++subset_size=1 ++seed=42 ++tags="['label-efficiency', 'NAIP_P_PRIOR', 'reproduce IPM']" &
# python train_baseline.py ++epoch=7 ++device=cuda:1 ++input_modality=NAIP_P_PRIOR ++test_cities="['durham_nc-2012_1m']" ++subset_size=1 ++seed=42 ++tags="['label-efficiency', 'NAIP_P_PRIOR', 'reproduce IPM']" &
# python train_baseline.py ++epoch=7 ++device=cuda:2 ++input_modality=NAIP_P_PRIOR ++test_cities="['austin_tx-2012_1m']" ++subset_size=1 ++seed=42 ++tags="['label-efficiency', 'NAIP_P_PRIOR', 'reproduce IPM']" &
# python train_baseline.py ++epoch=7 ++device=cuda:3 ++input_modality=NAIP_P_PRIOR ++test_cities="['phoenix_az-2010_1m']" ++subset_size=1 ++seed=42 ++tags="['label-efficiency', 'NAIP_P_PRIOR', 'reproduce IPM']" &

# python train_baseline.py ++epoch=14 ++device=cuda:0 ++input_modality=NAIP_ONLY ++test_cities="['pittsburgh_pa-2010_1m']" ++subset_size=0.5 ++seed=42 ++tags="['label-efficiency', 'NAIP_ONLY', 'reproduce IPM']" &
# python train_baseline.py ++epoch=14 ++device=cuda:2 ++input_modality=NAIP_ONLY ++test_cities="['durham_nc-2012_1m']" ++subset_size=0.5 ++seed=42 ++tags="['label-efficiency', 'NAIP_ONLY', 'reproduce IPM']" &
# python train_baseline.py ++epoch=14 ++device=cuda:2 ++input_modality=NAIP_ONLY ++test_cities="['austin_tx-2012_1m']" ++subset_size=0.5 ++seed=42 ++tags="['label-efficiency', 'NAIP_ONLY', 'reproduce IPM']" &
# python train_baseline.py ++epoch=14 ++device=cuda:3 ++input_modality=NAIP_ONLY ++test_cities="['phoenix_az-2010_1m']" ++subset_size=0.5 ++seed=42 ++tags="['label-efficiency', 'NAIP_ONLY', 'reproduce IPM']" &


# python train_baseline.py ++epoch=14 ++device=cuda:1 ++input_modality=NAIP_P_PRIOR ++test_cities="['pittsburgh_pa-2010_1m']" ++subset_size=0.5 ++seed=42 ++tags="['label-efficiency', 'NAIP_P_PRIOR', 'reproduce IPM']" &
# python train_baseline.py ++epoch=14 ++device=cuda:2 ++input_modality=NAIP_P_PRIOR ++test_cities="['durham_nc-2012_1m']" ++subset_size=0.5 ++seed=42 ++tags="['label-efficiency', 'NAIP_P_PRIOR', 'reproduce IPM']" &
# python train_baseline.py ++epoch=14 ++device=cuda:3 ++input_modality=NAIP_P_PRIOR ++test_cities="['austin_tx-2012_1m']" ++subset_size=0.5 ++seed=42 ++tags="['label-efficiency', 'NAIP_P_PRIOR', 'reproduce IPM']" &
# python train_baseline.py ++epoch=14 ++device=cuda:3 ++input_modality=NAIP_P_PRIOR ++test_cities="['phoenix_az-2010_1m']" ++subset_size=0.5 ++seed=42 ++tags="['label-efficiency', 'NAIP_P_PRIOR', 'reproduce IPM']" &

# sleep(30m)

# python train_baseline.py ++epoch=9 ++device=cuda:1 ++input_modality=NAIP_ONLY ++test_cities="['pittsburgh_pa-2010_1m']" ++subset_size=0.75 ++seed=42 ++tags="['label-efficiency', 'NAIP_ONLY', 'reproduce IPM']" &
# python train_baseline.py ++epoch=9 ++device=cuda:2 ++input_modality=NAIP_ONLY ++test_cities="['durham_nc-2012_1m']" ++subset_size=0.75 ++seed=42 ++tags="['label-efficiency', 'NAIP_ONLY', 'reproduce IPM']" &
# python train_baseline.py ++epoch=9 ++device=cuda:3 ++input_modality=NAIP_ONLY ++test_cities="['austin_tx-2012_1m']" ++subset_size=0.75 ++seed=42 ++tags="['label-efficiency', 'NAIP_ONLY', 'reproduce IPM']" &
# # python train_baseline.py ++epoch=9 ++device=cuda:3 ++input_modality=NAIP_ONLY ++test_cities="['phoenix_az-2010_1m']" ++subset_size=0.75 ++seed=42 ++tags="['label-efficiency', 'NAIP_ONLY', 'reproduce IPM']" &

# sleep(20m)

# python train_baseline.py ++epoch=9 ++device=cuda:0 ++input_modality=NAIP_P_PRIOR ++test_cities="['pittsburgh_pa-2010_1m']" ++subset_size=0.75 ++seed=42 ++tags="['label-efficiency', 'NAIP_P_PRIOR', 'reproduce IPM']" &
# python train_baseline.py ++epoch=9 ++device=cuda:2 ++input_modality=NAIP_P_PRIOR ++test_cities="['durham_nc-2012_1m']" ++subset_size=0.75 ++seed=42 ++tags="['label-efficiency', 'NAIP_P_PRIOR', 'reproduce IPM']" &
# python train_baseline.py ++epoch=9 ++device=cuda:2 ++input_modality=NAIP_P_PRIOR ++test_cities="['austin_tx-2012_1m']" ++subset_size=0.75 ++seed=42 ++tags="['label-efficiency', 'NAIP_P_PRIOR', 'reproduce IPM']" &
# # python train_baseline.py ++epoch=9 ++device=cuda:3 ++input_modality=NAIP_P_PRIOR ++test_cities="['phoenix_az-2010_1m']" ++subset_size=0.75 ++seed=42 ++tags="['label-efficiency', 'NAIP_P_PRIOR', 'reproduce IPM']" &

# sleep(20m)

python train_baseline.py ++epoch=20 ++device=cuda:1 ++input_modality=NAIP_ONLY ++test_cities="['pittsburgh_pa-2010_1m']" ++subset_size=0.35 ++seed=42 ++tags="['label-efficiency', 'NAIP_ONLY', 'reproduce IPM']" &
# python train_baseline.py ++epoch=20 ++device=cuda:2 ++input_modality=NAIP_ONLY ++test_cities="['durham_nc-2012_1m']" ++subset_size=0.35 ++seed=42 ++tags="['label-efficiency', 'NAIP_ONLY', 'reproduce IPM']" &
python train_baseline.py ++epoch=20 ++device=cuda:3 ++input_modality=NAIP_ONLY ++test_cities="['austin_tx-2012_1m']" ++subset_size=0.35 ++seed=42 ++tags="['label-efficiency', 'NAIP_ONLY', 'reproduce IPM']" &
# # python train_baseline.py ++epoch=20 ++device=cuda:3 ++input_modality=NAIP_ONLY ++test_cities="['phoenix_az-2010_1m']" ++subset_size=0.35 ++seed=42 ++tags="['label-efficiency', 'NAIP_ONLY', 'reproduce IPM']" &

# sleep(20m)

python train_baseline.py ++epoch=20 ++device=cuda:0 ++input_modality=NAIP_P_PRIOR ++test_cities="['pittsburgh_pa-2010_1m']" ++subset_size=0.35 ++seed=42 ++tags="['label-efficiency', 'NAIP_P_PRIOR', 'reproduce IPM']" &
# python train_baseline.py ++epoch=20 ++device=cuda:2 ++input_modality=NAIP_P_PRIOR ++test_cities="['durham_nc-2012_1m']" ++subset_size=0.35 ++seed=42 ++tags="['label-efficiency', 'NAIP_P_PRIOR', 'reproduce IPM']" &
python train_baseline.py ++epoch=20 ++device=cuda:2 ++input_modality=NAIP_P_PRIOR ++test_cities="['austin_tx-2012_1m']" ++subset_size=0.35 ++seed=42 ++tags="['label-efficiency', 'NAIP_P_PRIOR', 'reproduce IPM']" &
# # python train_baseline.py ++epoch=20 ++device=cuda:3 ++input_modality=NAIP_P_PRIOR ++test_cities="['phoenix_az-2010_1m']" ++subset_size=0.35 ++seed=42 ++tags="['label-efficiency', 'NAIP_P_PRIOR', 'reproduce IPM']" &
