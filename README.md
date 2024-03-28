# Stanford Ribonanza RNA Folding 32th place solution 

这个repository包含了我在kaggle上参加的Stanford Ribonanza RNA Folding竞赛中取得第32名的代码(排名前百分之四，获得银牌).  

Competiton website: [link](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding)  

本次比赛是一个RNA相关的seq_to_seq的预测. 从数据结构上来说, 其构成跟常规的NLP任务的输入一样每个输入数据均包含一段文本序列, 且序列中每个元素与其他位置的元素存在一定联系, 所以很适合用transfomer处理, 此外每个输入数据可能还有几个额外标量属性用于辅助预测, 需要模型自适应处理; 输出数据结构则与一般的NLP任务不一样, 其主要区别有2点区别: 1.此任务输入序列与输出序列长度相等且序列中每个元素一一对应; 2.我们需要模型预测的是序列上每个元素与两种固定反应之间的反应性强弱指标, 这是回归值, 而不是像输入一样的文本向量. 具体模型的输入输出数据说明,可详见上述链接中的介绍。  

我的解决方案基于一个encoder-decoder框架. 其中encoder采用的是根据本次任务改进的[Squeezeformer](https://arxiv.org/pdf/2206.00888.pdf), 相比一般的Transformer, 利用其attention-convolution框架可以更有效的提取局部和全局信息; 而decoder仅采用一个DNN模块, 因为我需要预测的并不是文本向量并不需要对其进行self-attention操作, 也可以减少causal padding等不必要的操作; 最后由于输入输出sequence一一对应, 所以模型预测时可以一次性生成整个序列的预测值, 而不需要像一般的sequenceto-sequence Transformer一样需要逐元素循环预测, 可以有效提升预测效率.  

同时为了有效提升模型泛化能力，特别是在私有测试数据序列平均长度相比训练数据长一倍的情况下, 我使用了以下正则化工具:
- 利用bpps计算得到的序列自相关矩阵作为pro_transform.  
- 对输出的各个回归值采用sigmoid运算. 
- 利用signal_to_noise，来修正的每个seq的在loss累加时的权重.

为了保证运行环境的一致性，我使用docker创建的container来部署模型, 所用image来自[此处](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch), 为了更高效的完成项目, 我使用jupyter notebook进行初期代码调试, 而后通过模块化处理将代码转移到python script上,以便更高效的进行超参数调节.训练时数据将实时上传保存至naptune.ai以方便我们对实验结果进行追踪可视化对比等.


## 准备


首先使用docker命令创建并运行container:
> docker run --gpus all --name 20240124  -it --shm-size=32g nvcr.io/nvidia/pytorch:23.07-py3  

下载此repository，并安装一些必要的packages  
> git clone https://github.com/johnzhangzzzz/Stanford-Ribonanza-RNA-Folding-32th-place-solution.git  
> cd Stanford-Ribonanza-RNA-Folding-32th-place-solution   
> pip install -r requirements.txt    
  
下载kaggle数据,  
> mkdir datamount  
> cd datamount

这里需要注意得设置kaggle登录口令才能使用kaggle命令下载训练数据, 如有需要口令可以邮件联系我zjhzjh124@icloud.com，或者自己注册一个kaggle账号  
> export KAGGLE_USERNAME="..."  
> export KAGGLE_KEY="..."  
> kaggle datasets download -d iafoss/stanford-ribonanza-rna-folding-converted  
> unzip -n stanford-ribonanza-rna-folding-converted.zip  
> rm stanford-ribonanza-rna-folding-converted.zip  
> cd ..  

其中的EternaFold库，需要需要编译下对应目录下Contrafold.cpp才能使用
> cd /workspace/Stanford-Ribonanza-RNA-Folding-32th-place-solution/repos/EternaFold/src  
> make contrafold  
> cd /workspace/Stanford-Ribonanza-RNA-Folding-32th-place-solution/

## model summary
显示模型的结构细节  
> python summary.py

## 训练  

> python train.py -C cfg_0
>> Explore the metadata in the Neptune app:  
>> <https://app.neptune.ai/common/quickstarts/e/QUI-99519/metadata>  
>> Checkpoint save : datamount/weights/cfg_0/fold0/checkpoint_last_seed2023.pth

本例中lr与loss的训练信息如下：
![fig_1.png](https://github.com/johnzhangzzzz/Stanford-Ribonanza-RNA-Folding-32th-place-solution/blob/72f1954835dc9bd4f3785bf48204e65d294be736/fig_1.png)     

模型训练结果，以及对应的训练参数均会上传至[neptune.ai](https://app.neptune.ai/common/quickstarts/e/QUI-99519/metadata)。




## 预测  
可将一个或者多个训练过模型的地址放入Checkpoint参数列表中，进行集成运算
> python infer.py --Checkpoint ['datamount/weights/cfg_0/fold0/checkpoint_last_seed2023.pth']




# References
SqueezeFormer (pytorch)[https://github.com/upskyy/Squeezeformer/](https://github.com/upskyy/Squeezeformer/)

