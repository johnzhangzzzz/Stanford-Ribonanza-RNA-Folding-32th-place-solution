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

BASEDIR= './'#'../input/asl-fingerspelling-config'
for DIRNAME in 'configs data models postprocess metrics utils repos'.split():
    sys.path.append(f'{BASEDIR}/{DIRNAME}/')

parser = argparse.ArgumentParser(description="")

parser.add_argument("-C", "--config", help="config filename", default="cfg_0")
parser.add_argument("-G", "--gpu_id", default="", help="GPU ID")
parser_args, other_args = parser.parse_known_args(sys.argv)
cfg = copy(importlib.import_module(parser_args.config).cfg)
#cfg = copy(importlib.import_module('cfg_0').cfg)

BPPs_RNA_Dataset = importlib.import_module(cfg.dataset).BPPs_RNA_Dataset
LenMatchBatchSampler = importlib.import_module(cfg.dataset).LenMatchBatchSampler
DeviceDataLoader = importlib.import_module(cfg.dataset).DeviceDataLoader
Squeezeformer_RNA = importlib.import_module(cfg.model).Squeezeformer_RNA
loss_f = importlib.import_module(cfg.loss).loss
MAE=importlib.import_module(cfg.metrics).MAE
OUT=cfg.OUT
SEED=cfg.SEED
nfolds=cfg.nfolds
seed_everything=importlib.import_module(cfg.utils).set_seed

seed_everything(SEED)
os.makedirs(OUT, exist_ok=True)

model = Squeezeformer_RNA(cfg,infer_mode='True').to(cfg.device)
RNA_Dataset_Test = importlib.import_module(cfg.dataset).RNA_Dataset_Test

import warnings 
warnings.filterwarnings("ignore")

df_test = pd.read_parquet(cfg.test)
df_test['L'] = df_test.sequence.apply(len)

gc.collect() 

MODELS = cfg.Checkpoint
models = []
for m in MODELS:
    model = Squeezeformer_RNA(cfg,infer_mode=True).to(cfg.device)
    model.load_state_dict(torch.load(m,map_location=torch.device('cpu'))['model'])
    model.eval()
    models.append(model)

ds = RNA_Dataset_Test(df_test)
dl = DeviceDataLoader(torch.utils.data.DataLoader(ds, batch_size=cfg.bs, 
               shuffle=False, drop_last=False, num_workers=cfg.num_workers), cfg.device)

ids,preds = [],[]
for x,y in tqdm(dl):
    with torch.no_grad(),torch.cuda.amp.autocast():
        p = torch.stack([torch.nan_to_num(model(x)['fc_outputs']) for model in models]
                        ,0).mean(0).clip(0,1)
        
    for idx, mask, pi in zip(y['ids'].cpu(), x['mask'].cpu(), p.cpu()):
        ids.append(idx[mask])
        preds.append(pi[mask[:pi.shape[0]]])

ids = torch.concat(ids)
preds = torch.concat(preds)

df = pd.DataFrame({'id':ids.numpy(), 'reactivity_DMS_MaP':preds[:,1].numpy(), 
                   'reactivity_2A3_MaP':preds[:,0].numpy()})

out_dir=f"{cfg.output_dir_sub}/fold{cfg.fold}/"

if not os.path.exists(out_dir): 
    os.makedirs(out_dir)

df.to_csv(f"{out_dir}/submission.csv", index=False, float_format='%.4f') # 6.5GB
df.head()
print(f"submission save : " +  f"{out_dir}/submission.csv")
