# Stanford Ribonanza RNA Folding 32th place solution 

这个repository包含了我在kaggle上参加的Stanford Ribonanza RNA Folding竞赛中取得第32名(排名前百分之四，获得银牌)的代码.  

Competiton website: [link](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding)  

简单来说，本次比赛本质是一个seq_to_seq的预测, 但不同于一般的asr任务（如翻译，归纳等）此任务输入序列与输出序列长度相等且两个序列中每个元素一一对应的。具体的说，输入数据跟常规的text输入结构相似，seq中每个元素与其他位置的元素存在联系，所以很适合用transfomer结构处理数据，不同点是每个seq还有几个额外属性，例如signal_to_noise等；输出与常见类型的asr任务不一样，我们需要模型计算输入序列上每个元素与两个固定反应类型之间的回归输出，即我们需要根据输入数据预测seq中每个元素对于反应A和反应B的反应性强弱。   

我主要的创新是： 
采用针对此次任务该进的Squeezeformer取代传统的transfomer作为主体结构。  
利用bpps计算得到的序列自相关矩阵，作为pro_transform, 可提升处理不同长度seq时的泛化能力。  

以下改动对在public测试集上的性能提升不大，但可有效提升在private测试机集上泛化能力，因为后者seq长度平均是前者的2倍：
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
其中的EternaFold库，需要需要编译下对应目录下Contrafold.cpp才能使用
> cd /workspace/Stanford-Ribonanza-RNA-Folding-32th-place-solution/repos/EternaFold/src  
> make contrafold
> cd /workspace/Stanford-Ribonanza-RNA-Folding-32th-place-solution/  

## 训练  

> python train.py -C cfg_0
>> Explore the metadata in the Neptune app:  
>> <https://app.neptune.ai/common/quickstarts/e/QUI-99519/metadata>  
>> Checkpoint save : datamount/weights/cfg_0/fold0/checkpoint_last_seed2023.pth
![fig_1.png](https://github.com/johnzhangzzzz/Stanford-Ribonanza-RNA-Folding-32th-place-solution/blob/72f1954835dc9bd4f3785bf48204e65d294be736/fig_1.png)    
模型训练结果，以及对应的训练参数均会上传至[neptune.ai](https://app.neptune.ai/common/quickstarts/e/QUI-99519/metadata)，

## model summary
显示模型的结构细节  
> python summary.py


## 预测  
可将一个或者多个训练的模型地址放入Checkpoint参数列表中
> python infer.py --Checkpoint ['datamount/weights/cfg_0/fold0/checkpoint_last_seed2023.pth']




# References
SqueezeFormer (pytorch)[https://github.com/upskyy/Squeezeformer/](https://github.com/upskyy/Squeezeformer/)
