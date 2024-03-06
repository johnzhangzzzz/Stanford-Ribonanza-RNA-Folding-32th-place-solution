# Stanford Ribonanza RNA Folding 32th place solution 

这个repository包含了我在kaggle上参加的Stanford Ribonanza RNA Folding竞赛取得第32名(排名前百分之四)并获得一个银牌的代码.本次比赛的核心，其实是一个seq_to_seq的预测,而且输入seq与输出seq是长度且一一对应的。输入跟常规的text输入有点类似，seq上每个向量都是存在联系的，不同点是每个序列还有几个额外属性，例如signal_to_noise等；输出与常见类型的asl任务不一样，我们需要模型计算输入序列上每个元素与两个固定反应类型之间的回归输出，即对于反应A和反应B，我们需要预测seq中每个元素对于反应A和反应B的反应性能，即回归输出。   
我主要的创新是，  
采用针对此次任务该进的Squeezeformer取代传统的transfomer结构。  
利用bpps计算得到的序列自相关矩阵，作为pro_transform, 可大大提升处理不同长度seq时的泛化能力。  

输出采用对输出采用sigmoid运算，此改动可提升在最终的测试数据泛化能力  
利用signal_to_noise，来修正的每个seq的在loss累加时的权重，此举提升了对最终测试数据的泛化能力  

# 准备

为了保持环境的一致性，我使用docker来部署模型,所用container来自[此处](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch),安装并运行container:
> docker run --gpus all --name 20240124  -it --shm-size=32g nvcr.io/nvidia/pytorch:23.07-py3  

下载此repository并安装必要的packages
> git clone https://github.com/johnzhangzzzz/Stanford-Ribonanza-RNA-Folding-32th-place-solution.git
> cd Stanford-Ribonanza-RNA-Folding-32th-place-solution  
> pip install -r requirements.txt  
  
下载kaggle数据,  
> mkdir datamount  
> cd datamount  
这里需要注意得设置kaggle登录口令才能在下载训练数据：如有需要口令可以邮件联系我zjhzjh124@icloud.com，或者自己注册一个kaggle账号  
> export KAGGLE_USERNAME="..."  
> export KAGGLE_KEY="..."  
> kaggle datasets download -d iafoss/stanford-ribonanza-rna-folding-converted  
> unzip -n stanford-ribonanza-rna-folding-converted.zip  
> rm stanford-ribonanza-rna-folding-converted.zip  
> cd ..
> 
EternaFold库包，需要需要编译下对应目录下Contrafold.cpp才能使用
> cd /workspace/Stanford-Ribonanza-RNA-Folding-32th-place-solution/repos/EternaFold/src  
> make contrafold
> cd /workspace/Stanford-Ribonanza-RNA-Folding-32th-place-solution/  

# 训练  
> python train.py -C cfg_0
>> Explore the metadata in the Neptune app:
>> https://app.neptune.ai/common/quickstarts/e/QUI-99276/metadata
>> Checkpoint save : datamount/weights/cfg_0/fold0/checkpoint_last_seed2023.pth

## 预测
> python infer.py --Checkpoint ['datamount/weights/cfg_0/fold0/checkpoint_last_seed2023.pth']




# References
SqueezeFormer (pytorch)[https://github.com/upskyy/Squeezeformer/](https://github.com/upskyy/Squeezeformer/)
