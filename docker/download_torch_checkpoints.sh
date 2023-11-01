#!/bin/bash

CKPT_PATH=$TORCH_HOME/hub/checkpoints
mkdir -p $CKPT_PATH

echo "Downloading WideResnet50_2 backbone"
curl --silent https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth -o $CKPT_PATH/wide_resnet50_2-95faca4d.pth  # Downloads Wide ResNet-50
echo "Downloading Resnet18 backbone"
curl --silent https://download.pytorch.org/models/resnet18-f37072fd.pth -o $CKPT_PATH/resnet18-f37072fd.pth   # Downloads ResNet-18
