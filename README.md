# Stanford Ribonanza RNA Folding 32th place solution 

这个repository包含了我在kaggle上参加的Stanford Ribonanza RNA Folding竞赛中取得第32名的代码(排名前百分之四，获得银牌).  

Competiton website: [link](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding)  

本次比赛是一个RNA相关的seq_to_seq的预测. 从数据结构上来说, 其构成跟常规的NLP任务的输入一样每个输入数据均包含一段文本序列, 且序列中每个元素与其他位置的元素存在一定联系, 所以很适合用transfomer处理, 此外每个输入数据可能还有几个额外的标量指标用于辅助预测(类似于ChatGPT中聊天机器人的性格设置参数), 需要模型自适应处理; 输出数据结构则与一般的NLP任务不一样, 其主要区别有2点区别: 1.此任务输入序列与输出序列长度相等且序列中每个元素一一对应; 2.我们需要模型预测的是序列上每个元素与两种固定反应之间的反应性强弱指标, 这是回归值, 而不是像输入一样的文本向量. 具体模型的输入输出数据说明,可详见上述比赛链接中的介绍。  

我的解决方案基于一个encoder-decoder框架. 其中encoder采用的是针对本次任务改进的Squeezeformer, 利用其attention-convolution框架可以更有效的提取局部和全局信息, 具体结构细节可查看[此论文](https://arxiv.org/pdf/2206.00888.pdf); 而decoder仅采用一个DNN模块, 因为我需要预测的并不是文本向量并不需要对其进行self-attention操作; 最后由于输入输出sequence一一对应, 所以模型预测时可以一次性生成整个序列的预测值, 而不需要像一般的sequence-to-sequence Transformer一样需要利用causal padding等工具逐元素循环叠加预测, 可以有效提升预测效率. 

同时为了有效提升模型泛化能力，特别是在私有测试数据序列平均长度相比训练数据长一倍的情况下, 我使用了以下正则化工具:
- 利用bpps计算得到的序列自相关矩阵作为pro_transform.  
- 对输出的各个回归值采用sigmoid运算. 
- 利用signal_to_noise，来修正的每个seq的在loss累加时的权重.

此ML项目基于pytorch实现, 其中包含数据预处理, 模型构筑与训练, 实验数据追踪等模块. 为了保证运行环境的一致性，我使用docker创建的container来部署模型, 所用image来自[此处](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch). 同时为了更高效的完成项目, 我使用jupyter notebook进行初期代码调试, 而后通过模块化处理将代码转移到python script上, 以便更高效的进行超参数调节. 训练时数据将实时上传保存至naptune.ai以方便我们对实验结果进行追踪可视化对比等.


## 准备


（可选，推荐在虚拟系统上部署此项目，防止一些安装冲突）首先使用docker命令创建并运行container  
> docker run --gpus all --name 20240124  -it --shm-size=32g nvcr.io/nvidia/pytorch:23.07-py3  

进入创建的虚拟系统命令行界面，下载并进入此repository
> git clone https://github.com/johnzhangzzzz/Stanford-Ribonanza-RNA-Folding-32th-place-solution.git  
> cd Stanford-Ribonanza-RNA-Folding-32th-place-solution

安装packages
> pip install -r requirements.txt    
  
下载kaggle数据,  
> mkdir datamount  
> cd datamount  
> kaggle datasets download -d iafoss/stanford-ribonanza-rna-folding-converted  
> unzip -n stanford-ribonanza-rna-folding-converted.zip  
> rm stanford-ribonanza-rna-folding-converted.zip  
> cd ..  

其中的EternaFold库，需要需要编译下对应目录下Contrafold.cpp才能使用
> cd /workspace/Stanford-Ribonanza-RNA-Folding-32th-place-solution/repos/EternaFold/src  
> make contrafold  
> cd /workspace/Stanford-Ribonanza-RNA-Folding-32th-place-solution/

## model summary
模型可视化:  
> python summary.py

运行结果如下:
![fig_2.png](https://github.com/johnzhangzzzz/Stanford-Ribonanza-RNA-Folding-32th-place-solution/blob/main/fig_2.png)

## 训练  

> python train.py -C cfg_0
>> Explore the metadata in the Neptune app:  
>> <https://app.neptune.ai/common/quickstarts/e/QUI-99519/metadata>  
>> Checkpoint save : datamount/weights/cfg_0/fold0/checkpoint_last_seed2023.pth

训练数据会实时同步至至neptune.ai, 用于数据可视化  
例如, 若要查看本次训练相关的数据[可点击上述链接](https://app.neptune.ai/common/quickstarts/e/QUI-99519/metadata), 从中可以看到本例中lr与loss的训练信息如下：
![fig_1.png](https://github.com/johnzhangzzzz/Stanford-Ribonanza-RNA-Folding-32th-place-solution/blob/72f1954835dc9bd4f3785bf48204e65d294be736/fig_1.png)     

程序运行完成后模型被保存为datamount/weights/cfg_0/fold0/checkpoint_last_seed2023.pth  


## 预测  
可将一个或者多个训练过模型的地址放入Checkpoint参数列表中，进行集成预测
> python infer.py --Checkpoint ['datamount/weights/cfg_0/fold0/checkpoint_last_seed2023.pth']


# References
EternaFold: [https://github.com/eternagame/EternaFold](https://github.com/eternagame/EternaFold)   
arnie: [https://github.com/DasLab/arnie](https://github.com/DasLab/arnie)  
SqueezeFormer (pytorch): [https://github.com/upskyy/Squeezeformer/](https://github.com/upskyy/Squeezeformer/)

