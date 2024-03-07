import os
import pandas as pd
import os, gc
import numpy as np
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

import neptune
from neptune.utils import stringify_unsupported
from tqdm import tqdm, notebook
import transformers
from collections import defaultdict
import glob

import sys
import argparse
from copy import copy
import importlib

from torchinfo import summary

BASEDIR= './'#'../input/asl-fingerspelling-config'
for DIRNAME in 'configs data models postprocess metrics utils repos'.split():
    sys.path.append(f'{BASEDIR}/{DIRNAME}/')

parser = argparse.ArgumentParser(description="")

parser.add_argument("-C", "--config", help="config filename", default="cfg_0")
parser.add_argument("-G", "--gpu_id", default="", help="GPU ID")
parser_args, other_args = parser.parse_known_args(sys.argv)
cfg = copy(importlib.import_module(parser_args.config).cfg)

df = pd.read_parquet(cfg.train_df)
LenMatchBatchSampler = importlib.import_module(cfg.dataset).LenMatchBatchSampler
DeviceDataLoader = importlib.import_module(cfg.dataset).DeviceDataLoader
Squeezeformer_RNA = importlib.import_module(cfg.model).Squeezeformer_RNA
BPPs_RNA_Dataset = importlib.import_module(cfg.dataset).BPPs_RNA_Dataset

fold=cfg.fold
nfolds=cfg.nfolds

ds_train = BPPs_RNA_Dataset(df, mode='train', fold=fold, nfolds = nfolds)
ds_train_len = BPPs_RNA_Dataset(df, mode='train', fold=fold, 
            nfolds=nfolds, mask_only=True)
sampler_train = torch.utils.data.RandomSampler(ds_train_len)
len_sampler_train = LenMatchBatchSampler(sampler_train, batch_size=cfg.bs,
            drop_last=True)
dl_train = DeviceDataLoader(torch.utils.data.DataLoader(ds_train,
                                                        batch_sampler=len_sampler_train,
                                                        num_workers=cfg.num_workers,
                                                        persistent_workers=True),
                                                        cfg.device)

sample =next(iter(dl_train))[0]

Squeezeformer_RNA = importlib.import_module(cfg.model).Squeezeformer_RNA
model = Squeezeformer_RNA(cfg,infer_mode='True').to(cfg.device)

tensors = {'inputs':sample['inputs'],'input_lengths':sample['input_lengths'],'seq':sample['seq']}
summary(
    model,
    input_data=[tensors],
    col_names=["input_size", "output_size", "num_params", "trainable"],
    col_width=25,
    depth=4
)

y=model(sample);
y['fc_outputs'];

#!apt-get update
#!apt-get install graphviz
#!pip install torchviz
#make_dot(y['fc_outputs'])
#make_dot(y['fc_outputs'], params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")