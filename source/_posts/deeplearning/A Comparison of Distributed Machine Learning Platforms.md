---
title: A Comparison of Distributed Machine Learning Platforms
date: 2020-02-14 20:44:24
tags: introduce
category: deeplearning
---

2017年发表的一篇论文《A Comparison of Distributed Machine Learning Platforms》，针对各分布式平台的测试

# Three Pass

## The first pass
**列出**
1. 标题，摘要，介绍
 > **标题**：A Comparison of Distributed Machine Learning Platforms，顾名思义，分布式平台的比较，究竟谁快谁慢
 > **摘要**：现在是大数据和大计算量时代，所谓大，就是一个集群已经不能满足运行了，所以才会有分布式平台出现，论文开始调研各种分布式机器学习平台，描述平台的架构组成，并比较**性能，扩展性**：其中spark是dataflow类型系统，PMLS是参数服务器系统，tensorflow和mxnet是优化得dataflow系统，主要对比这四种平台系统。**我们从分布式系统的角度，分析了这些方法的通信和控制瓶颈。**做实验的条件是：mnist数据集的逻辑回归分类
 > **介绍**：分析为什么会有分布式，还是相似的原因，数据量大，模型使用更多数据，效果更佳。
 **DataFlow System：MapReduce，Spark，Naiad，使用a directed graph构建业务流程，**这些针对简单的机器学习可行
 **Parameter-server System：Google DistBelief，PMLS**针对复杂模型，可更新大参数量的模型
 **Advanced DataFlow System：Google TensorFlow，mxnet，**这些数据流系统允许具有可变状态的循环图，并且可以模拟参数服务器的功能。
 本论文的贡献点：比较全面比较不同分布式机器学习平台，实验结果表明PS结构最好，但只是针对CPU的，tf和mxnet的能力未体现。
2. 章节
 > SPARK DATAFLOW SYSTEM
 > PMLS PARAMETER-SERVER SYSTEM
 > TENSORFLOW
 > MXNET: ref7
 > EVALUATION:Ganglia监控COU、网络、内存使用的工具ref3
 > CONCLUDING REMARKS
3. 数学原理
 > 无
4. 结论
 > 
5. 文献
 > Ganglia分布式监控资源工具
**回答**
1. 论文类型
 > 平台系统测试与对比
2. 相关内容
 > 分布式
3. 正确性
 > 
4. 贡献点
 > 
5. 相关文献
 >
6. 是否继续
 > 不建议继续
## The second pass
**图表**

**文献背景**

## The third pass
**实验**

**重现**
