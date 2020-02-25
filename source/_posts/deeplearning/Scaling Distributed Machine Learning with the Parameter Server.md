---
title: Scaling Distributed Machine Learning with the Parameter Server
date: 2020-02-12 20:44:24
tags: introduce
category: [deeplearning,distributedTraining]
---
李沐大神2014年发表的关于mxnet深度学习框架如何实现分布式训练，分布式训练未来是一个趋势，时间成本的节约能大大加速模型评估，让更多人能抽出时间深入到算法，而不是局限于调参。

# Three Pass

## the first pass
**列出**
1. 标题，摘要，介绍
 - 标题：Scaling Distributed Machine Learning with the Parameter Server，基于参数服务器形式的大规模分布式机器学习
 - 摘要：第一眼没太看懂，作者发现参数服务器框架的分布式问题，数据与工作流都分布到各个工作节点，而参数服务器节点主要维护参数。框架提供节点异步通信，模型一致，容错性，可扩展。并且为了说明还跑了pb级别的真实数据。（只是将这个原理实现了，并实际运行了LDA和Distributed Sketching算法）
 - 介绍：要做到大规模分布式训练需要分布式优化和推断，这对于数据超大和模型参数超大都是难点，如何解决实现更快速的训练时重点。所面临的三个问题：
 > 参数传递耗费网络带宽多
 > 很多算法是序列的，会造成同步和延迟时间长
 > 分布式，容错性得强，机器可能不可靠，工作可能被抢占
 论文介绍PS结构的实现学术界与工业界都很多，本文主要关注**分布式推理**部分，
 **参数pull和push非常多，参数服务器提供了一种有效的手段来更新参数：每个参数节点只维护一部分参数（参数服务器可是并行化？所以每个worker只从一个参数服务器取一部分参数）**
 > **通信**：不用常用的K-V对形式，而是将参数化成向量形式传递
 > **容错**：
 第一代是K-V对形式，第二代是DistBelief
2. 章节
> Machine Learning: Goals, Risk Minimization, Generative Models
> Architecture: (Key, Value)Vectors, Range Push and Pull, User-Defined Functions on the Server, Asynchronous Tasks and Dependency, Flexible Consistency, User-defined Filters
> Implementation: Vector Clock, Messages, Consistent Hashing, Replication and Consistency, Server Management, Worker Management
> Evaluation: Sparse Logistic Regression, Latent Dirichlet Allocation, Sketches
> Summary and Discussion
> References

3. 数学原理
 > 实验的LR、LDA模型有数学部分

4. 结论
 > 1、简单使用，2、收敛快，3、容错、可扩展，第三代分布式机器学习

5. 文献
文献43是第一代：An architecture for parallel topic models. In Very Large Databases (VLDB),2010.

**回答**
1. 论文类型
 > 原型系统介绍与测试
2. 相关内容
 > 分布式训练：TensorFlow、pytorch、mxnet、horovod
3. 正确性
 > 在2014年算是分布式才刚刚起步，到现在可能horovod可以替代
4. 贡献点、创新点
 > 罗列出了分布式所面临的困难，并重点关注通信与容错，提供了一个实现框架可供使用
5. 相关文献

6. 是否继续
 > 先放放

## the second pass
**图表**

**相关文献背景**

## the third pass
**重现**

**实验**