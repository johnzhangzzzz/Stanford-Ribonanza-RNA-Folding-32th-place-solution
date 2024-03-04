# Stanford Ribonanza RNA Folding  

这个repository包含了我在kaggle上参加的Stanford Ribonanza RNA Folding竞赛取得第32名(排名前百分之4)并获得一个银牌的代码,
其中的模型用于预测RNA分子结构和由此产生的化学图谱

我的模型主要利用了一个针对次任务的而改良的Squeezeformer,并引入了arnie,进行

# 准备

为了更方便的在不同环境下运行程序,我使用了虚拟环境docker,
首先安装镜像:
> docker run --gpus all --name 20240124  -it --shm-size=32g nvcr.io/nvidia/pytorch:23.07-py3
下载库,

下载数据
...

训练


# References
SqueezeFormer (pytorch)[https://github.com/upskyy/Squeezeformer/](https://github.com/upskyy/Squeezeformer/)
