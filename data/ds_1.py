import os
import pandas as pd
import os, gc
import numpy as np
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

#%env PYTHONPATH=/workspace/Stanford-Ribonanza-RNA-Folding-32th-place-solution/arnie
os.environ['PYTHONPATH']='/workspace/Stanford-Ribonanza-RNA-Folding-32th-place-solution/repos/arnie'
#%env ARNIEFILE=/workspace/Stanford-Ribonanza-RNA-Folding-32th-place-solution/my_arnie_file.txt
os.environ['ARNIEFILE']='/workspace/Stanford-Ribonanza-RNA-Folding-32th-place-solution/repos/arnie/my_arnie_file.txt'
#'/workspace/Stanford-Ribonanza-RNA-Folding-32th-place-solution/my_arnie_file.txt'
#%env ETERNAFOLD_PATH=/workspace/Stanford-Ribonanza-RNA-Folding-32th-place-solution/EternaFold/src
os.environ['ETERNAFOLD_PATH']='/workspace/Stanford-Ribonanza-RNA-Folding-32th-place-solution/repos/EternaFold/src'
#%env ETERNAFOLD_PARAMETERS=/workspace/Stanford-Ribonanza-RNA-Folding-32th-place-solution/EternaFold/parameters/EternaFoldParams.v1
os.environ['ETERNAFOLD_PARAMETERS']='/workspace/Stanford-Ribonanza-RNA-Folding-32th-place-solution/repos/EternaFold/parameters/EternaFoldParams.v1'

from arnie.bpps import bpps
my_sequence = 'CGCUGUCUGUACUUGUAUCAGUACACUGACGAGUCCCUAAAGGACGAAACAGCG'
sss=bpps(my_sequence, package='eternafold')


  
         
def dict_to(x, device='cuda'):
    '''对字典中的数据使用to(device)方法
    zz={'a':1,'b':2}
    for z in zz:
        print(zz[z])
    '''
    return {k:x[k].to(device) for k in x}

'''def to_device(x, device='cuda'):
    return tuple(dict_to(e,device) for e in x)'''

class DeviceDataLoader:
    '''
    先对dataloader迭代产生的数据，包括df&target，应用dict_to方法，封装到tuple中
    '''
    def __init__(self, dataloader, device='cuda'):
        self.dataloader = dataloader
        self.device = device
        #self.n=1
    
    def __len__(self):
        return len(self.dataloader)
    
    def __iter__(self):
        for batch in self.dataloader:
            yield tuple(dict_to(x, self.device) for x in batch)

class BPPs_RNA_Dataset(Dataset):
    '''
    将原始数据包装成torch.utils.data.Dataset类，
    并使得输出数据满足Squeezeformer输入需求
    '''
    def __init__(self, df, mode='train', seed=2023, fold=0, nfolds=4, 
                 mask_only=False, **kwargs):
        self.seq_map = {'A':0,'C':1,'G':2,'U':3}
        self.Lmax = 206
        #self.zzz= BPPS()
        df['L'] = df.sequence.apply(len)
        df_2A3 = df.loc[df.experiment_type=='2A3_MaP']
        df_DMS = df.loc[df.experiment_type=='DMS_MaP']
        
        split = list(KFold(n_splits=nfolds, random_state=seed, 
                shuffle=True).split(df_2A3))[fold][0 if mode=='train' else 1] #分成4个fold
        df_2A3 = df_2A3.iloc[split].reset_index(drop=True)
        df_DMS = df_DMS.iloc[split].reset_index(drop=True)
        
        m = (df_2A3['SN_filter'].values > 0) & (df_DMS['SN_filter'].values > 0)
        df_2A3 = df_2A3.loc[m].reset_index(drop=True)
        df_DMS = df_DMS.loc[m].reset_index(drop=True)
        
        self.seq = df_2A3['sequence'].values
        self.L = df_2A3['L'].values
        
        self.react_2A3 = df_2A3[[c for c in df_2A3.columns if \
                                 'reactivity_0' in c]].values
        self.react_DMS = df_DMS[[c for c in df_DMS.columns if \
                                 'reactivity_0' in c]].values
        self.react_err_2A3 = df_2A3[[c for c in df_2A3.columns if \
                                 'reactivity_error_0' in c]].values
        self.react_err_DMS = df_DMS[[c for c in df_DMS.columns if \
                                'reactivity_error_0' in c]].values
        self.sn_2A3 = df_2A3['signal_to_noise'].values
        self.sn_DMS = df_DMS['signal_to_noise'].values
        self.mask_only = mask_only
        
    def __len__(self):
        return len(self.seq)  
    
    def __getitem__(self, idx):
        seq = self.seq[idx]
        #seq0=seq
        if self.mask_only:
            mask = torch.zeros(self.Lmax, dtype=torch.bool)
            mask[:len(seq)] = True
            return {'mask':mask},{'mask':mask}
        bpp = bpps(seq, package='eternafold')
        bpp = torch.tensor(bpp,dtype=torch.float32)
        #bpp=bpp.float()# torch.tensor(bpp,dtype=torch.float32)
        p2d=(0,(self.Lmax-len(bpp)),0,(self.Lmax-len(bpp)))
        bpp = torch.nn.functional.pad(bpp,p2d,"constant", 0)

        seq = [self.seq_map[s] for s in seq]
        seq = np.array(seq)###,dtype=np.float32
        mask = torch.zeros(self.Lmax, dtype=torch.bool)
        mask[:len(seq)] = True
        input_lengths=torch.tensor(len(seq),dtype=torch.int32)
        seq = np.pad(seq,(0,self.Lmax-len(seq)))
        
        react = torch.from_numpy(np.stack([self.react_2A3[idx],
                                           self.react_DMS[idx]],-1))
        react_err = torch.from_numpy(np.stack([self.react_err_2A3[idx],
                                               self.react_err_DMS[idx]],-1))
        sn = torch.FloatTensor([self.sn_2A3[idx],self.sn_DMS[idx]])
        
        return {'inputs':torch.from_numpy(seq), 'input_lengths':input_lengths, 'seq':bpp}, \
               {'react':react, 'react_err':react_err,
                'sn':sn, 'mask':mask}
    
class RNA_Dataset_Test(Dataset):
    '''将数据包装成infer时需要的输入格式'''
    def __init__(self, df, mask_only=False, **kwargs):
        self.seq_map = {'A':0,'C':1,'G':2,'U':3}
        #df['L'] = df.sequence.apply(len)
        self.Lmax = df['L'].max()
        self.df = df
        self.mask_only = mask_only
        
    def __len__(self):
        return len(self.df)  
    
    def __getitem__(self, idx):
        id_min, id_max, seq = self.df.loc[idx, ['id_min','id_max','sequence']]
        mask = torch.zeros(self.Lmax, dtype=torch.bool)
        L = len(seq)
        mask[:L] = True
        if self.mask_only: return {'mask':mask},{}
        ids = np.arange(id_min,id_max+1)
        bpp = bpps(seq, package='eternafold')
        bpp = torch.tensor(bpp,dtype=torch.float32)
        p2d=(0,(self.Lmax-L),0,(self.Lmax-L))
        bpp = torch.nn.functional.pad(bpp,p2d,"constant", 0)
        
        seq = [self.seq_map[s] for s in seq]
        seq = np.array(seq)
        input_lengths=torch.tensor(len(seq),dtype=torch.int32)
        seq = np.pad(seq,(0,self.Lmax-L))
        ids = np.pad(ids,(0,self.Lmax-L), constant_values=-1)
        
        return {'inputs':torch.from_numpy(seq), 'input_lengths':input_lengths,'mask':mask,'seq':bpp}, \
               {'ids':ids}
    

class LenMatchBatchSampler(torch.utils.data.BatchSampler):
    '''由于每个sample中mask token数量不一，所以长度匹配采样，让每个batch中的实际训练数据量保持一致'''
    def __iter__(self):
        buckets = [[]] * 100
        yielded = 0

        for idx in self.sampler:
            s = self.sampler.data_source[idx]
            if isinstance(s,tuple): L = s[0]["mask"].sum()
            else: L = s["mask"].sum()
            L = max(1,L // 16) 
            if len(buckets[L]) == 0:  buckets[L] = []
            buckets[L].append(idx)
            
            if len(buckets[L]) == self.batch_size:
                batch = list(buckets[L])
                yield batch
                yielded += 1
                buckets[L] = []
                
        batch = []
        leftover = [idx for bucket in buckets for idx in bucket]

        for idx in leftover:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yielded += 1
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yielded += 1
            yield batch
            
