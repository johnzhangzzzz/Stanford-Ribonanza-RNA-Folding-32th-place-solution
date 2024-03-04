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

BASEDIR= './' 
for DIRNAME in 'configs data models postprocess metrics utils repos'.split():
    sys.path.append(f'{BASEDIR}/{DIRNAME}/')

parser = argparse.ArgumentParser(description="")
parser.add_argument("-C", "--config", help="config filename", default="cfg_0")
parser.add_argument("-G", "--gpu_id", default="", help="GPU ID")
parser_args, other_args = parser.parse_known_args(sys.argv)
cfg = copy(importlib.import_module(parser_args.config).cfg)

if parser_args.gpu_id != "":
    os.environ['CUDA_VISIBLE_DEVICES'] = str(parser_args.gpu_id)

# overwrite params in config with additional args
if len(other_args) > 1:
    other_args = {k.replace('-',''):v for k, v in zip(other_args[1::2], other_args[2::2])}

    for key in other_args:
        if key in cfg.__dict__:

            print(f'overwriting cfg.{key}: {cfg.__dict__[key]} -> {other_args[key]}')
            cfg_type = type(cfg.__dict__[key])
            if cfg_type == bool:
                cfg.__dict__[key] = other_args[key] == 'True'
            elif cfg_type == type(None):
                cfg.__dict__[key] = other_args[key]
            else:
                cfg.__dict__[key] = cfg_type(other_args[key])

#start naptun
fns = [parser_args.config] + [getattr(cfg, s) for s in 'dataset model'.split()]#获取文件名
fns = sum([glob.glob(f"{BASEDIR }/*/{fn}.py") for fn in  fns], [])#获取文件的相对路径

if cfg.neptune_project == "common/quickstarts":
    neptune_api_token=neptune.ANONYMOUS_API_TOKEN
else:
    neptune_api_token=cfg.neptune_api_token
    #os.environ['NEPTUNE_API_TOKEN']

neptune_run = neptune.init_run(
        project=cfg.neptune_project,
        tags="demo_0",
        mode="async",
        api_token=neptune_api_token,
        capture_stdout=False,
        capture_stderr=False,
        #source_files=fns
    )

#print(f"Neptune system id : {neptune_run._sys_id}")
#print(f"Neptune URL       : {neptune_run.get_url()}")
neptune_run["cfg"] = stringify_unsupported(cfg.__dict__)

df = pd.read_parquet(cfg.train_df)

BPPs_RNA_Dataset = importlib.import_module(cfg.dataset).BPPs_RNA_Dataset
LenMatchBatchSampler = importlib.import_module(cfg.dataset).LenMatchBatchSampler
DeviceDataLoader = importlib.import_module(cfg.dataset).DeviceDataLoader
Squeezeformer_RNA = importlib.import_module(cfg.model).Squeezeformer_RNA
loss_f = importlib.import_module(cfg.loss).loss
MAE=importlib.import_module(cfg.metrics).MAE
OUT=cfg.OUT
SEED=cfg.SEED
nfolds=cfg.nfolds
fold=cfg.fold
set_seed=importlib.import_module(cfg.utils).set_seed

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
###torch.utils.data.DataLoader：Data loader combines a dataset and a sampler, and provides an iterable over the given dataset.
###对未进行DeviceDataLoader类进行包装的ds_train使用iter()方法后，属性变成MultiProcessingDataLoaderIter，主要原因ds_train是返回对象为两个{}，
###所以通过使用DeviceDataLoader，类进行包装可以让两个{}变为一个

ds_val = BPPs_RNA_Dataset(df, mode='eval', fold=fold, nfolds=nfolds)
ds_val_len = BPPs_RNA_Dataset(df, mode='eval', fold=fold, nfolds=nfolds, 
            mask_only=True)
sampler_val = torch.utils.data.SequentialSampler(ds_val_len)
len_sampler_val = LenMatchBatchSampler(sampler_val, batch_size=cfg.bs, 
            drop_last=False)
dl_val= DeviceDataLoader(torch.utils.data.DataLoader(ds_val, 
            batch_sampler=len_sampler_val, num_workers=cfg.num_workers), cfg.device)

if not os.path.exists(f"{cfg.output_dir}/fold{fold}/"): 
    os.makedirs(f"{cfg.output_dir}/fold{fold}/")

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
model = Squeezeformer_RNA(cfg).to(cfg.device)

total_steps = len(ds_train)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.warmup * (total_steps // cfg.bs),
            num_training_steps=cfg.epochs * (total_steps // cfg.bs),
            num_cycles=0.5
        ) #设置学习率变化函数
scaler = GradScaler() #实例化梯度缩放类，用于防止梯度爆炸

import warnings 
warnings.filterwarnings("ignore")
#由于使用了混合精度，

# Start the training and validation loop
cfg.curr_step = 0 #
optimizer.zero_grad()
total_grad_norm = None    
total_grad_norm_after_clip = None
i = 0 

L_dl_train = len(dl_train)
L_dl_val = len(dl_val)
if cfg.debug:
    L_dl_train = L_dl_train//cfg.debug_fact
    L_dl_val = L_dl_val//cfg.debug_fact
    
for epoch in range(cfg.epochs):
    cfg.curr_epoch = epoch
    progress_bar = tqdm(range(L_dl_train)[:], desc=f'Train epoch {epoch}')
    tr_it = iter(dl_train)
    losses = []
    gc.collect()
    
    model.train()
    for itr in progress_bar:
        i += 1
        cfg.curr_step += cfg.bs
        data = next(tr_it)
        torch.set_grad_enabled(True)
        batch=data
        if cfg.mixed_precision:
            with autocast():
                output_dict = model(batch)
        else:
            output_dict = model(batch)
        loss = output_dict["loss"]
        losses.append(loss.item())

        if cfg.grad_accumulation >1: 
            loss /= cfg.grad_accumulation
        #有时候内存不够，batchsize太小的时候，用这种方法等效的增大batchsize

         #以下为利用梯度混合训练设置，原理详见   
         #https://zhuanlan.zhihu.com/p/165152789 和 https://pytorch.org/docs/stable/amp.html
            
        if cfg.mixed_precision:
            scaler.scale(loss).backward()

            if i % cfg.grad_accumulation == 0:
                if (cfg.track_grad_norm) or (cfg.clip_grad > 0): #吴恩达的视频中有介绍，梯度归一&梯度裁剪 防止梯度爆炸的技术
                    scaler.unscale_(optimizer)                          
                if cfg.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if i % cfg.grad_accumulation == 0:
                if cfg.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                optimizer.step()
                optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        loss_names = [key for key in output_dict if 'loss' in key]
        for l in loss_names:
            neptune_run[f"train/{l}"].log(value=output_dict[l].item(), step=cfg.curr_step)

        neptune_run["lr"].log(
                value=optimizer.param_groups[0]["lr"], step=cfg.curr_step
            )
        if total_grad_norm is not None:
            neptune_run["total_grad_norm"].log(value=total_grad_norm.item(), step=cfg.curr_step)
            neptune_run["total_grad_norm_after_clip"].log(value=total_grad_norm_after_clip.item(), step=cfg.curr_step)           

    #if (epoch + 1) % cfg.eval_epochs == 0 or (epoch + 1) == cfg.epochs: 
    if 1>0:           
        model.eval()
        torch.set_grad_enabled(False)#also,torch.inference_mode()
        val_data = defaultdict(list) #若字典检索不到key，不会报错而回返回一个空的list
        val_score = 0 #本次任务的评价函数与损失函数均为MAE，所以没有额外定义metrics

        progress_bar = tqdm(range(L_dl_val)[:], desc=f'Val epoch {epoch}')
        tr_it = iter(dl_val)

        #for ind_, data in enumerate(tqdm(dl_val, desc=f'Val epoch {epoch}')):
            #batch = batch_to_device(data, cfg.device)
            #batch = data
        
        for itr in progress_bar:
            data = next(tr_it)
            batch=data

            if cfg.mixed_precision:
                with autocast():
                    output_dict_val = model(batch)
            else:
                output_dict_val = model(batch)
                #单个batch的输出
            #常规的计算输出
            for key, val in output_dict_val.items():
                val_data[key] += [output_dict_val[key]]

        for key, val in output_dict_val.items():
            value = val_data[key]
            if isinstance(value[0], list):
                val_data[key] = [item for sublist in value for item in sublist]
            else:
                if len(value[0].shape) == 0:
                    val_data[key] = torch.stack(value)
                else:
                    val_data[key] = torch.cat(value, dim=0) 
     
        #平铺每个batch的输出,并累计所有的batch，用于计算metrics
        #val_data['fc_outputs'].shape=torch.Size([45082, 206, 2])，val_data['loss'].shape=torch.Size([1409])，   
        if cfg.save_val_data:
            torch.save(val_data, f"{cfg.output_dir}/fold{fold}/val_data_seed{SEED}.csv")

         

        #val_df = val_dataloader.dataset.df
            
        #pp_out = post_process_pipeline(cfg, val_data, val_df)

        #val_score = calc_metric(cfg, pp_out, val_df, "val")
        loss_names_val = [key for key in output_dict_val if 'loss' in key]
        loss_names_val += [key for key in output_dict_val if 'score' in key]       
        
        val_score =val_data['loss'].mean()
        if type(val_score)!=dict:
            val_score = {f'score':val_score}

                   
        
        for k, v in val_score.items():
            print(f"val_{k}: {v:.3f}")
            if neptune_run:
                neptune_run[f"val/{k}"].log(v, step=epoch)       
    
    if not cfg.save_only_last_ckpt:
        torch.save({"model": model.state_dict()}, f"{cfg.output_dir}/fold{fold}/checkpoint_last_seed{cfg.SEED}.pth")    
        


    
torch.save({"model": model.state_dict()}, f"{cfg.output_dir}/fold{fold}/checkpoint_last_seed{cfg.SEED}.pth")
print(f"Checkpoint save : " +  f"{cfg.output_dir}/fold{fold}/checkpoint_last_seed{cfg.SEED}.pth")

run_id = neptune_run["sys/id"].fetch()
neptune_run.stop()