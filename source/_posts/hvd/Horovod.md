---
title: Horovod fast and easy distributed deep learning in TensorFlow
date: 2020-02-05 22:44:24
tags: [deepLearning,distributedTraining]
category: introduce
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
【1】图表说明
原生TensorFlow分布式在GPU数量增加时，每秒处理图片数量却不是线性增加，且GPU越多，浪费越多，安利一个GPU一秒处理200张图片，2个如果在线性增加的情况下应该能处理400张，所以TensorFlow分布式训练效果不好，耗费时间长的原因等待分析(GPU型号使用的都是一样的，生产环境也容易出现不同能力的GPU，那么整个集群能力受限于计算能力最弱的GPU)
![](/images/horovod/1.png "tensorflow分布式在GPU数量增加时的表现")

进行分布式训练的方式，常用时数据并行，每个GPU都有相同的模型，但处理的数据不同，之后再将训练更新的参数汇总，这对应PS架构的参数服务器
![](/images/horovod/2.png "tensorflow分布式在GPU数量增加时的表现")

PS架构是有中心化节点的一种结构，模型参数存放在中心节点，当有更新后都会下发到各个worker节点，类似Hadoop的master/slave结构，但其集群容错性较好，如果是多中心节点，即使一个挂了，其他还能运行（这点根据实验，中心节点挂一个就会阻塞，所以集群还是得用其他方式来进行容错设计，比如zookeeper）。
而且PS结构在分发参数时，如果模型参数量接近带宽总量（总worker的参数量超过带宽就会发生），分发就不是并行的，容易串行分发，增加了通信时长，这也是其GPU数量增加，计算能力却未线性增加的主要原因。总次数(2*worker数)
此外，通信量总是worker数与模型参数的总量之积。
![](/images/horovod/3.png "tensorflow分布式在GPU数量增加时的表现")

PS结构通信耗费时间长，horovod基于ring all-reduce算法，去中心化构建分布式训练集群，形成一个有向环，每个worker分配一份参数进行更新，主要两步：
1、reduce：worker将自己的一份参数发给下个worker，一次更新会迭代(worker数-1)次
2、gather：将所有更新的参数更新到每个worker，一次更新会迭代(worker数-1)次
相比PS结构，更新次数相对较少，每次传输的参数量只是整个模型的一部分
![](/images/horovod/4.png "tensorflow分布式在GPU数量增加时的表现")

horovod提供的timeline分析工具，是TensorFlow自带的一个封装，可以清晰看到整个训练时间内，哪些地方耗时多。
![](/images/horovod/5.png "tensorflow分布式在GPU数量增加时的表现")

使用horovod做的对比实验，可以看出horovod基本趋于线性增加的计算能力当GPU数量增加，未能完全一样，可能GPU数量多了，通信耗时成本也在增加，但整体好于TensorFlow(有个问题就是不清楚该次实验是否也是以GPU作为一个worker的级别，大概率是这样，但也有可能是服务器为worker的级别)
![](/images/horovod/6.png "tensorflow分布式在GPU数量增加时的表现")

本次实验室对比horovod依赖不同通信框架的结果，基于TCP的是依赖openMPI，而RDMA是另外一种直接内存读取的通信手段，各有利弊。基于RDMA的horovod效果还是优于TCP(未去分析不同模型的效果，因一些模型在分布式训练本身就有劣势，具体待分析)
![](/images/horovod/7.png "tensorflow分布式在GPU数量增加时的表现")

【2】相关文献
1、Accurate, large minibatch SGD: Training ImageNet in 1 hour, 2017, arXiv:1706.02677.
2、Tensorflow: Large-scale machine learning on heterogeneous distributed systems, 2016, arXiv:1603.04467
3、Tensorflow: A system for large-scale machine learning, 2016, arXiv:1605.08695
4、Bandwidth optimal all-reduce algorithms for clusters of workstations. J.Parallel Distrib. Comput., 69:117–124, 2009
5、[Bringing HPC Techniques to Deep Learning](http://www.gibiansky.com/)

## The third pass
【1】论文的假设：当能极力优化通信时间，GPU数量增加与集群计算能力成线性增长
【2】重现论文实验：需要两台服务器，4张GPU显卡，一般都没有如此环境使用，要么科研、要么公司、要么云端环境(不清楚是否可自定义安装)，该篇论文也确实带来了福利，在模型越来越大，数据越来越多，训练一个模型能越快越好
【3】论文比对的是TensorFlow，还有其他框架pytorch、mxnet、caffe等未比较，可实验对比

More info: [Deployment](https://hexo.io/docs/one-command-deployment.html)
