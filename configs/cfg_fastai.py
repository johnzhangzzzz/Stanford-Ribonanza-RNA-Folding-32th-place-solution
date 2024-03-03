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

#logging
cfg.neptune_project = "zjh/RNA"
cfg.neptune_connection_mode = "async"
cfg.tags = "base"
cfg.neptune_api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjZTQ1ODM1YS1jZjI2LTQxYzItYjkxZC1jYWZjYTI3MzNjNGIifQ=="

cfg.Checkpoint=[
    'datamount/weights/cfg_0/fold0/checkpoint_last_seed2023.pth',
    ]
cfg.train_df= f'datamount/train_data.parquet'
cfg.test = f'datamount/test_sequences.parquet'
cfg.debug=True
cfg.debug_fact=10
cfg.output_dir = f"datamount/weights/{os.path.basename(__file__).split('.')[0]}" #__file__是python中的内置变量，表示当前文件的文件名
cfg.dataset = "ds_1"
cfg.bs = 32
cfg.num_workers = 16
cfg.device= 'cuda' if torch.cuda.is_available() else 'cpu'
cfg.model='models_fastai'
cfg.loss='loss'
cfg.metrics = 'metrics'
cfg.OUT = './'
cfg.SEED=2023
cfg.nfolds=4
cfg.fold=0
cfg.utils='utl'
cfg.epochs = 2
cfg.mixed_precision=True
cfg.grad_accumulation = 8.
cfg.lr=1e-6
cfg.weight_decay = 0.05
cfg.clip_grad = 4.
cfg.track_grad_norm=False
cfg.warmup = 10

#EVAL
cfg.calc_metric = True
cfg.eval_epochs = 1
cfg.save_val_data = False
cfg.training_test = True
# augs & tta

#Saving
cfg.save_weights_only = True
cfg.save_only_last_ckpt = True

#model
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