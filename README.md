# Bicameral Neural Network

## Overview
Left–Right brain model with cross-attention (corpus callosum).

## Install
pip install torch

## Run
python bicameral_model.py

## 2-line run
from bicameral_model import BicameralModel; import torch
print(BicameralModel(10)(torch.randn(1,10)))
