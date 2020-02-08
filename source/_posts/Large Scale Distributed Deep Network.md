---
title: Large Scale Distributed Deep Networks
date: 2020-02-08 20:44:24
tags: introduce
category: [deeplearning,distributedTraining]
---
2012年Google提出的一篇关于分布式训练。

# Three pass

## first pass
**列出**
1. 标题，摘要，介绍
标题：Large Scale Distributed Deep Networks，大规模分布式深度网络
摘要：分布式训练好处，论文实验，主要还是针对模型参数量大的进行实验，使用了异步随机梯度下降和Sandblaster [L-BFGS](https://www.hankcs.com/ml/l-bfgs.html)在Google第一代深度学习框架DistBelief完成，只是在ImageNet验证是可行方案
介绍：现行的深度学习为什么需要分布式训练，**模型越大或数据越多，模型的准确率就越好，那么大模型或大数据量下，单机训练模型不容易，所以需要分布式训练**；而深度学习分布式需要关注的是梯度如何计算，也就是SGD的变种，如何适应集群，论文只是在多cpu下进行实验，2012年GPU还未像现在这么流行。论文还有两个结果：1、相同模型，相同准确率下，分布式训练速度比单个GPU快10倍，这是收敛速度；2、在ImageNet数据集训练10亿参数量的模型，刷新SOTA结果

2. 章节
> Previoous work
> Model parallelism
> Distributed oprimization algorithms: Downpour SGD,Sandblaster L-BFGS
> Experiments
> Conclusions
> References
> Appendix:SGD, L-BFGS的算法介绍

3. 数学原理
分布式理论，只有Downpour SGD有对应数学原理，可先不看

4. 结论
这篇论文其实应该算DistBelief的简单介绍，并实现Downpour SGD和Sandblaster L-BFGS，在2000多个CPU上实验

5. 相关文献
文献相对较老，也算是一些在图片语音上进行分布式训练尝试的文章

**回答**
1. 论文类型
现存系统DistBelief介绍

2. 相关背景论文
相关的比如：图片分布式训练这类的

3. 正确性


4. 贡献点
DistBelief框架，TensorFlow的前身
Downpour SGD和Sandblaster L-BFGS实现

5. 条理
条理清晰，起码在讲如何实现一个分布式训练，算法都给定了

6. 是否继续
不建议继续，但Downpour SGD和Sandblaster L-BFGS这两个算法之后可以继续深入
并且论文做了模型并行实验，这块可以了解

## second pass
**图表**

**文献**

## third pass
**重现**
 
