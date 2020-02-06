---
title: Bringing HPC Techniques to Deep Learning
date: 2020-02-06 20:44:24
tags: framework
category: deeplearning
---
这是一篇博客[Bringing HPC Techniques to Deep Learning](http://www.gibiansky.com/)，介绍在百度使用ring all reduce技术，实现大规模使用多GPU计算能力

# Three pass

## first pass
**列出**
【1】标题：Bringing HPC Techniques to Deep Learning，引入高可用的计算技术到深度学习，表明是提供一种解决性能的技术，训练效率
介绍：一些背景，自从2012年计算机视觉比赛上引入卷积神经网络起，6亿的参数在两张GPU上训练花费一周时间，到2016年32张GPU训练10亿参数的模型花费三周时间，说明1)模型越来越大，训练耗时越来越长，2)多GPU共同训练一个模型的技术越来越重要，百度SVAIL实验室实现了多GPU训练的技术
【2】章节标题：The Communication Problem（多个GPU不在一个服务器上，如何通信），The Ring Allreduce（GPU间通信算法），The Scatter-Reduce，The Allgather，Allreduce Communication Cost（理论上Ring Allreduce算法在通信上的耗时），Applying the Allreduce to Deep learning，conclusion，reference
【3】数学原理：无
【4】结论：ring all reduce算法有效平均梯度在多个设备上，并且比较在数据并行和同步更新上更易实现，且罗列了可用的ring all reduce开源库：[baidu-ring all-reduce](https://github.com/baidu-research/baidu-allreduce)，[Uber horovod](https://github.com/horovod/horovod) 
【5】参考文献
1. ImageNet classification with deep convolutional neural networks. Advances in neural information processing systems. 2012. 
2. Exploring the limits of language modeling. arXiv preprint arXiv:1602.02410 (2016).

**回答**
【1】论文类型：原型研究描述，ring all-reduce算法的实现
【2】相关论文：horovod
【3】正确性：
【4】贡献点：百度有开源库，Uber也有，并详细介绍了该算法的原理
【5】条理：针对当前背景的需要，详细介绍了一种可用的技术

## second pass
**图表**
【1】基本
最基本的基于中心节点的PS结构，10个GPU的话，有一个GPU专门作为参数汇总与分发的节点，说明只剩9个GPU进行模型前向和后向的计算，如果带宽是1.2千兆字节的，模型3百万个参数，每个参数是float类型占4个字节，总共模型有1千兆字节，一秒只能传输一个GPU的，也就是一次迭代10个GPU耗时9秒（之后的模型都是要统计参数量，再乘以float类型的字节数就是模型总量）

【2】ring all-reduce
ring all-reduce是一种带宽最优化的算法，以环形构建集群，每个节点只与前后相邻节点有通信，减少了PS结构中参数服务器节点的工作量，并分发到其他每个节点。
![](/images/HPC/1.png)
该算法有两部分：
1. scatter-reduce：GPU间相互交换各自块的数据，来计算本轮最终的参数结果,**该部分主要是将所有GPU计算的结果进行汇总，汇总后每个GPU都有一块数据是最终的结果**
2. all-gather：将最终的结果进行交换，**将前面汇总后的数据分发到其他未更新的块，达到所有块都是最终的结果**

【3】通信花销
N个GPU，发送（scatter-reduce）参数`N-1`次，接收（all-gather）参数`N-1`次，每次每个GPU传输的数据量为`K/N`，其中K表示模型参数总量，所以总的传输量为：
```
2(N-1)·K/N
```
而PS结构N个GPU，总次数`2(N-1)`，总传输量`2(N-1)K`，从传输数据量也能看出，ring all-reduce小于PS结构，如果K越接近带宽最大值，ring all-reduce通信耗时将远远小于PS（PS结构又是all-reduce结构）

**相关文献背景**

## third pass
**重现**
 
名词：
1. gigabytes
2. GPUDirect RDMA
3. Infiniband 
