---
title: Horovod: fast and easy distributed deep learning in TensorFlow
date: 2020-02-05 20:44:24
tags: introduce
category: DeepLearning, distributed training
---
业务导向，最近在看深度学习分布式训练框架内容，虽然每类深度学习框架都提供了分布式训练的接口，但总有利弊，所有才会有其他分布式深度学习框架。本文是对论文【Horovod: fast and easy distributed deep learning in TensorFlow】的了解。

# #Three pass approach
基于上一篇论文阅读技巧，进行三步走，每步如下：
## The first pass
索引 | 问题 | 内容 | 说明
:-: | :-: | :-: | :-: 
1	| 标题 | 标题 | Horovod: fast and easy distributed deep learning in TensorFlow|
	| 摘要 | 摘要 | 提高多GPU利用率的库 |
	| 介绍 | 介绍 | 当前背景 |
2   | 章节 | 标题 | Introduction、 Going distributed、 Leveraging a different type of algorithm、Introducing Horovod、Distributing your training job with Horovod、 Horovod Timeline、Tensor Fusion、 Horovod Benchmarks、 Next steps、|
    | 小节 | 标题 | NO |
3   | 数学内容 | 原理框架 | 2(N-1)轮 |
4   | 结论 | 结论 | 开源该库，效果比TensorFlow分布式好 |
5   | 引用 | 引用 | TensorFlow分布式相关和框架使用到的NCCL、OpenMPI、RDMA等技术 |

···回答如下几个问题
【1】论文类别：研究原型描述，并测试对比现有TensorFlow分布式
【2】上下文关联：TensorFlow、pytorch、mxnet等深度学习框架的PS分布式训练与Ring all-reduce结构
【3】正确性：对比实验都假设在增加GPU数量时，计算能力应该呈线性增长，隐藏
【4】创新点：引用ring all-reduce结构，减少参数通信时间已达到GPU数量与计算能力呈线性增长，并开源该技术。
【5】论文条理：从章节标题可以看出，horovod在基于什么背景下，解决了什么问题，并实验验证其结论的正确性
基于以上问题回答，该论文可以深入了解

## The second pass

## The third pass


More info: [Deployment](https://hexo.io/docs/one-command-deployment.html)
