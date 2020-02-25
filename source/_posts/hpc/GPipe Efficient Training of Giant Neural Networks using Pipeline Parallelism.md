---
title: GPipe Efficient Training of Giant Neural Networks using Pipeline Parallelism
date: 2020-02-20
tags: introduce
category: [deepLearning,distributedTraining,model parallel]
---

Google在2019年发表的一篇关于模型并行训练的文章，毕竟现在是拼大模型大数据量的时代，手动滑稽，33rd Conference on Neural Information Processing Systems (NeurIPS 2019), Vancouver, Canada

# Three pass

## The First Pass
**列出**
1. 标题，摘要，介绍
	- 标题：GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism, 实用的pipeline并行方案，使用不一样的模型并行来训练
	- 摘要：已经被证明过的，大规模深度网络模型具有更好的质量，但单纯提高模型网络容量，受限于单个机器容量，所以基本使用特殊算法和结构来实现，即灵活性不足，只适用当前任务。主要为了能高效进行模型并行，开发了GPipe库，以pipeline形式将网络层序列化。作者也做了两个实验，1、Image Classification，2、Multilingual Neural Machine Translation，两个实验的模型参数量都很大
	- 介绍：为什么需要模型并行-模型越来越大，现有模型并行方案-基本是任务式，除了这个任务其他不能使用，最终解决方案-新的模型并行库，GPipe，如何来解决-1、将模型分成具有顺序的层(按层划分？)，2、连续的层可划分为一个cell，每个cell放在不同的加速器上，3、使用batch splitting技术，将mini-batch划分为更小的单元micro-batch，4、之后通过pipeline在每个cell中执行micro-batch数据(还是顺序执行吗，还是每个cell执行一部分数据)，5、以同步更新梯度方式训练；这样解决效果如何-两个实验效果都不错
2. 章节
	- 2 The GPipe Library: Interface, Algorithm, Performance Optimization
	- 3 Performance Analyses: Performance Overhead Breakdown
	- 4 Image Classification
	- 5 Massive Massively Multilingual Machine Translation
	- 6 Design Features and Trade-Offs
	- 7 Conclusion
	- References

3. 数学原理
	- 开发的库，类似框架类，无数学原理

4. 结论
	- 作者提供的GPipe库以同步更新梯度的方式，并对batch进行划分来达到模型并行，未知是否可以用在多个机器之间，不单单是单机，可以看看实验用了什么设备，几个设备。
	- 作者还对大规模卷积和基于transformer的模型做了实验，效果都不错。
	- 作者认为GPipe具有三个特性：
	1 高效性：提出的batch splitting方法使GPipe能达到线性加速，在设备增长情况下
	2 灵活性：GPipe支持任何顺序的神经网络
	3 可靠性：同步更新梯度，与划分分区无关
	- 总结起来跟我们现在业务需要是一致的，我们也想任何网络都是模型并行，并且能扩展到多机多卡上

5. 文献
	- 可能大模型现在在NLP领域比较常见，CV领域比较常见的就是类别比较多时，比如亿万级人脸识别，这不是单机多卡就能完成模型并行训练的，必须扩展到多机多卡，如果能虚拟化GPU设备当然更好
	- [16] Lingvo: a modular and scalable framework for sequence-to-sequence modeling. arXiv preprint arXiv:1902.08295, 2019.
	- 文献18是关系mxnet的，这篇已看过，看来他们也在选框架，Mxnet: A flexible and efficient machine learning library for heterogeneous distributed systems. arXiv preprint arXiv:1512.01274, 2015
	- [34] Mesh-tensorflow: Deep learning for supercomputers. In Neurips, pages 10414–10423, 2018
	- [40]  On model parallelization and scheduling strategies for distributed machine learning. In Neurips, pages 2834–2842, 2014.
	- [45] Pipedream: Fast and efficient pipeline parallel dnn training. arXiv preprint arXiv:1806.03377, 2018.

**回答**
1. 论文类型
	- 技术开创，基于现有网络开发的更高效框架 
	
2. 相关内容
	- 模型并行方案，Pipedream，Mesh-tensorflow

3. 正确性
	- 该第三方库已开源，基于论文实验结论是有效果的

4. 贡献点
	- 提供了一个通用模型并行框架，并开源

5. 参考文献
	- 已读：
	- 未读：

6. 是否继续
	- 继续，需要了解GPipe如何进行模型并行，基于什么语言框架实现，是否支持多机多卡，使用体验如何，代码侵入严重不

## The Second Pass
**图表**
1. 第一章
	- 背景介绍，只要引出通用模型并行框架开发的必要性，实验结果表明GPipe是可用的

2. 第二章
	介绍GPipe的特性与执行原理
	![](/images/GPipe/1.png "GPipe")
	- 开源，基于Lingvo框架实现，可在caffe、mxnet、pytorch框架下实现核心功能
	- 所有的深度神经网络多可用定义为L层的序列。用户需要指定：
	1、模型划分几块
	2、micro-batch大小
	3、
	- 算法：原始的模型并行是将模型按层划分k份，分别放到k个GPU上，GPU之间执行是串行的，现在将mini-batch也划分未k份，第一份跑完后给第二个GPU，那么就实现了并行，第一个GPU跑第二份，第二个GPU跑第一份，跑完的参数进行累积，并往后传？
3. 第三章
	- 性能分析：对使用GPipe和不使用GPipe模型参数量分析，8个加速器下可以有25倍容量提升，对人脸识别是否也可以用
	- 实验环境：两个实验，，主要考虑scalability、efficiency、communication cost，设备考虑了1、2、4、8个加速器，可以认为最多8个GPU，单机多卡(NVIDIA P100 GPU)
	1、AmoebaNet Convolutional Model：输入224x224，mini-batch 128,1）不使用GPipe可以训82M参数，2）
	2、Transformer seq-seq Model
	主要还是提高计算效率的，原生的模型并行，同一时刻只有一个GPU在计算
4. 第四章：Image Classification
	- 将ImageNet输入大小扩大为480x480进行训练，划分4个partition(即4个GPU)，效果有提升
	- 主要点就是对卷积网络的模型并行
5. 第五章：Massive Massively Multilingual Machine Translation
	- NLP领域的实验
6. 第六章：Design Features and Trade-Offs
	分析现有模型并行系统
	- 模型并行核心：划分网络到不同设备
	- 现有系统问题：设备利用率低，通信容易达到瓶颈
	- 主要两种方式来解决：1、Single Program Multiple Data(SPMD)，2、pipeline parallelism
	- 现存解决工具：
	1、SPMD思路：运行划分每个计算到多个设备上，通信消耗大，需细化
		Mesh-TensorFlow：主要用于数据并行
	2、pipeline思路：主要减少通信耗时
		PipeDream：
		GPipe：需单个层固定在单个加速器上，即只能按层划分，BatchNorm支持需特殊设计

**相关文献**

## The Third Pass
**实验**

**重现**