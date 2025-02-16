# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

# CelebA
DATASET="celeba"
DIR="/home/${USER}/deepcluster/datasets/celeba/train"
ARCH="alexnet"
LR=0.05
WD=-5
K=10000
WORKERS=12
SOBEL=false
EXP="/home/${USER}/deepcluster/trained_models/"
RESUME="/home/${USER}/deepcluster/trained_models/deepcluster_celeba_checkpoint.pth.tar"

# NORB
# DATASET="norb"
# DIR="/home/${USER}/deepcluster/datasets/norb/train"
# ARCH="alexnet"
# LR=0.005
# WD=-5
# K=1000
# WORKERS=0 # for some reason, with WORKERS being non zero, threading got block when loading batches from dataloader
# SOBEL=false
# EXP="/home/${USER}/deepcluster/trained_models/"
# RESUME="/home/${USER}/deepcluster/trained_models/deepcluster_norb_checkpoint.pth.tar"

# Causal3D
# DATASET="causal3d"
# DIR="/home/${USER}/deepcluster/datasets/causal3d"
# ARCH="alexnet"
# LR=0.005
# WD=-5
# K=1000
# WORKERS=0 # for some reason, with WORKERS being non zero, threading got block when loading batches from dataloader
# SOBEL=false
# EXP="/home/${USER}/deepcluster/trained_models/"
# RESUME="/home/${USER}/deepcluster/trained_models/deepcluster_causal3d_checkpoint.pth.tar"

# MPI3D
# DATASET="mpi3d"
# DIR="/home/${USER}/deepcluster/datasets/mpi3d_dataset/train" #NOTE: for previous res, seemingly using the entire mpi3d dataset for training deepcluster
# ARCH="alexnet"
# LR=0.005
# WD=-5
# K=1000
# WORKERS=12
# SOBEL=false
# EXP="/home/${USER}/deepcluster/trained_models/"
# RESUME="/home/${USER}/deepcluster/trained_models/deepcluster_mpi3d_checkpoint.pth.tar"

mkdir -p ${EXP}

python main.py --dataset ${DATASET} --datadir ${DIR} --exp ${EXP} --arch ${ARCH} --resume ${RESUME} \
  --lr ${LR} --wd ${WD} --k ${K} --sobel ${SOBEL} --verbose --workers ${WORKERS} --checkpoints 10000
