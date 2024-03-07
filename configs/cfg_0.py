import os
import sys
from importlib import import_module
import platform
import json
import numpy as np
import torch
import pandas as pd
from transformers.models.speech_to_text import Speech2TextConfig
#import augmentations as A
from types import SimpleNamespace

cfg = SimpleNamespace(**{})

#neptune_logging
#cfg.neptune_project = "zjh/RNA"
cfg.neptune_project = "common/quickstarts"
cfg.neptune_connection_mode = "async"
cfg.tags = "base"
cfg.neptune_api_token="..."

#path of data&model
cfg.Checkpoint=[
    'datamount/weights/cfg_0/fold0/checkpoint_last_seed2023.pth',
    ]
cfg.train_df= f'datamount/train_data.parquet'
cfg.test = f'datamount/test_sequences.parquet'
cfg.output_dir = f"datamount/weights/{os.path.basename(__file__).split('.')[0]}" #__file__是python中的内置变量，表示当前文件的文件名
cfg.output_dir_sub = f"datamount/submission/{os.path.basename(__file__).split('.')[0]}"
cfg.dataset = "ds_1"
cfg.loss='loss'
cfg.metrics = 'metrics'
cfg.model='transfomer_block'
cfg.utils='utl'
cfg.OUT = './'

#data set
cfg.debug=True #debug模式，用少量数据方便调试
cfg.debug_fact=10 #实际数据=总数据//debug_fact
cfg.nfolds=4 #总的fold数
cfg.fold=0 #当前fold
cfg.bs = 32 #data batchsize
cfg.num_workers = 16 #预处理数据时cpu线程
cfg.device= 'cuda' if torch.cuda.is_available() else 'cpu'
cfg.SEED=2023

#train_set
cfg.epochs = 30
cfg.mixed_precision=True
cfg.grad_accumulation = 8.
cfg.lr=1e-6
cfg.weight_decay = 0.05
cfg.clip_grad = 4.
cfg.track_grad_norm=False
cfg.warmup = 10

#EVAL
#cfg.calc_metric = True
#cfg.eval_epochs = 1
#cfg.training_test = True
# augs & tta

#Saving
cfg.save_val_data = False
cfg.save_weights_only = True
cfg.save_only_last_ckpt = True

#model参数配置
cfg.num_classes=1
encoder_config = SimpleNamespace(**{})
encoder_config.input_dim = 80#80
encoder_config.encoder_dim = 384#384
encoder_config.num_encoder_layers = 16#16
encoder_config.reduce_layer_index= 70
encoder_config.recover_layer_index = 150
encoder_config.num_attention_heads = 8#8
encoder_config.feed_forward_expansion_factor = 4
encoder_config.conv_expansion_factor = 2
encoder_config.input_dropout_p = 0.1
encoder_config.feed_forward_dropout_p = 0.1
encoder_config.attention_dropout_p = 0.1
encoder_config.conv_dropout_p = 0.1
encoder_config.conv_kernel_size = 31
encoder_config.half_step_residual = False
cfg.encoder_config = encoder_config