# Stanford Ribonanza RNA Folding  

这个repository包含了我在kaggle上参加的Stanford Ribonanza RNA Folding竞赛取得第32名(排名前百分之四)并获得一个银牌的代码,
其中的模型用于预测RNA分子结构和由此产生的化学图谱

我的模型主要利用了一个针对次任务的而改良的Squeezeformer,并引入了arnie,进行

# 准备

为了更方便的在不同环境下运行程序,我使用了虚拟环境docker,  

## 首先安装镜像:
> docker run --gpus all --name 20240124  -it --shm-size=32g nvcr.io/nvidia/pytorch:23.07-py3  

## 下载库
> cd Stanford-Ribonanza-RNA-Folding-32th-place-solution  
> pip install -r requirements.txt  
> mkdir datamount  
> cd datamount  
> 
## 下载kaggle数据
首先需要设置kaggle登录口令用于下载训练数据：如有需要口令可以邮件联系我zjhzjh124@icloud.com
> export KAGGLE_USERNAME="..."  
> export KAGGLE_KEY="..."  
> kaggle datasets download -d iafoss/stanford-ribonanza-rna-folding-converted  
> unzip -n stanford-ribonanza-rna-folding-converted.zip  
> rm stanford-ribonanza-rna-folding-converted.zip  
> cd ..  
> cd /workspace/Stanford-Ribonanza-RNA-Folding-32th-place-solution/repos/EternaFold/src  
> make contrafold  


...

## 训练  
> python train.py -C cfg_0
>> Explore the metadata in the Neptune app:
>> https://app.neptune.ai/common/quickstarts/e/QUI-99276/metadata
>> Checkpoint save : datamount/weights/cfg_0/fold0/checkpoint_last_seed2023.pth

## 预测
> python infer.py --Checkpoint ['datamount/weights/cfg_0/fold0/checkpoint_last_seed2023.pth']




# References
SqueezeFormer (pytorch)[https://github.com/upskyy/Squeezeformer/](https://github.com/upskyy/Squeezeformer/)
