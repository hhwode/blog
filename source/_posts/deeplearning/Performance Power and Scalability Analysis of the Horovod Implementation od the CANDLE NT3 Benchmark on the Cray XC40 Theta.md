---
title: Performance Power and Scalability Analysis of the Horovod Implementation od the CANDLE NT3 Benchmark on the Cray XC40 Theta
date: 2020-02-13 20:44:24
tags: introduce
category: deeplearning
---
芝加哥大学Xingfu Wu等于2018年发表的一篇horovod框架的测试论文

# Three Pass

## The first pass
**列出**
1. 标题，摘要，介绍
 > 标题：Performance Power and Scalability Analysis of the Horovod Implementation od the CANDLE NT3 Benchmark on the Cray XC40 Theta，其中有几个词不清楚是什么：CANDLE NT3 Benchmark应该是某种模型，Cray XC40 Theta这是超级计算机的一种型号，也就是验证horovod的性能，功率和扩展性
 > 摘要：主要分析的是NT3模型的可扩展性、性能和功率在不同batch size和学习率下的影响，通过节点功率文件，CPU，内存和timeline来进行分析horovod，并且在NT3增加节点数量时，horovod通信开销可以hold住。实验结论也表明batch size增加可以减少训练时间，学习率变大可以更快收敛，并轻微减少功率使用。还有一些遗留问题。
 > 介绍：前置条件介绍
  【1】分布式工具使用horovod，而不是原生tensorflow代码，也介绍了为什么选用horovod，开源，代码修改少，分布式比原生tensorflow快接近2倍在每秒处理图片数上
  【2】CANDLE project：使用单个DNN代码解决三个问题
  【3】NT3：keras实现的一种CANDLE project代码
  【4】文献15的一个用 Horovod-like Cray CPE ML Plugin比horovod和tensorflow分布式效果都好，但不开源，并且只针对 Cray systems，所以只能使用horovod
  【5】horovod并行实现关注NT3的： scalability, performance, and power characteristics在不同batch size和学习率下的两种内存模式，cache和flat，在Cray XC40 Theta环境上
  【6】实验表明：1、cache模式比flat模式开销更小，2、batch size增加可减少训练时间和功率消耗，3、增加学习率可以轻微减少训练时间和功率，并增加准确率
  【7】论文贡献如下：1、使用horovod实现的NT3模型可以作为参考，用于实现其他模型，2、分析了NT3基准测试的Horovod实现的可伸缩性，并讨论了Horovod**开销**，3、研究了Horovod实现的NT3基准测试的性能和功率特性，并使用功率分析来分析学习速率和批大小等参数如何影响性能和功率（什么是power profiling）
  【8】后面章节围绕几个名词：NT3，horovod，Cray XC40 Theta，实验，分析
  
2. 章节
 > CANDLE NT3 Benchmark and Its Horovod Implementation: CANDLE NT3 Benchmark, Horovod, Using Horovod to Parallelize the NT33 Benchmark
 > System Platform: Cray XC40 Theta
 > Scalability Analysis: Original NT3 Benchmark under Different Memory Modes, Horovod NT3 Benchmark
 > Performance and Power Analysis of the Horovod NT3 Benchmark: Cache Mode, Flat Mode
 > Conclusions
 > Reference

3. 数学原理
 > 如何用，如何针对某个问题的实现，如数学原理

4. 结论
 >  论文本身关注什么：performance，power，scalability；这些因素在调节batch size和学习率和两种内存模式-cache、flat mode的影响
 > 关注的东西在实验分析后得出：batch size增大，训练时间减少，其实就是分布式规模可扩展，并且能提高训练速度
 > 遗留问题：1、数据加载时间，2、horovod数据shard和shuffle，可减少loss提高acc，3、使用的cProfile工具只包含总的tf运行时间，没有细节，可供选择工具NVProf

5. 文献

**回答**
1. 论文类型
 > 现有系统测试，使用horovod实现了现有业务需要的DNN代码，并进行了对应性能、功率和扩展性的测试
 
2. 相关背景内容
 > 可以说是对horovod的一种测试，里面实验方式可供参考

3. 正确性
 > 从结论上看，batch size增加能减少训练时间，与分布式原理一致

4. 贡献点、创新点
 > 1、比较系统的分析了horovod的性能，2、实验方式可供参考，3、实验分析方法可借鉴

5. 文献
 >  Scaling Deep Learning, ALCF SDL(Simulation, Data and Learning) Workshop, March 2018
 >  [NVProf](https://docs.nvidia.com/cuda/profiler-users-guide/index.html)
 >  [Python Profilers](https://docs.python.org/2/library/profile.html)

6. 是否继续
 > 建议继续深入，主要关注实验方式和实验分析方法

## The second pass
**图表**
- 第二章：介绍NT3和horovod
 NT3：就是一维卷积，涉及部分1、data loading,2、preprocessing，3、basic training and cross-validation，4、prediction and evaluation on test data；训练集597MB，测试集149MB，SGD优化器，400个epochs，batch size是20，learning rate是0.001
 Horovod：hvd的目标是使分布式深度学习更快更简单，hvd核心概念是基于MPI概念的，比如size、rank、local rank、allreduce、allgather、broadcast；Horovod的一个独特的特性是它能够交错通信和计算；使用MPI_Allreduce()来平均梯度
- 第三章：Cray XC40 Theta系统平台介绍
 Cray XC40 Theta：64 compute cores(一个英特尔Phi骑士登陆(KNL) 7230与热设计功率(TDP) 215瓦)，16gb的高带宽内包内存，192GB的DDR4 RAM，128GB的SSD，共享L2缓存32 MB(由两个核心共享的1MB L2缓存)，使用210GB/s的带宽，剩下是专门针对Cray系统的一些监控软件
 cache mode优先分配内存是MCDRAM，之后才是DDR
 ![cache mode](/images/Performance/1.png "cache mode")
 flat mode优先分配内存是DDR4，之后是MCDRAM
 ![flat mode](/images/Performance/2.png "flat mode")
 不同的mode对性能和功率都有影响，所以才会研究cache mode和flat mode
- 第四章：扩展性分析
 【1】对比cache mode和flat mode在单机一个epoch下功率消耗，结果是cache mode相对较好，刚好能缓存所有数据，并且batch size越大，功率消耗越低，因限制没有无限尝试，毕竟单机内存等有限
 【2】单个epoch，batch size为20进行horovod实验，通过cProfile来记录每个操作的耗时，一个epoch是过完所有训练集，所以即使集群节点不同，处理的数据量也是一样的，因为有节点间通信开销，所以节点越多，训练相同数据量耗时越长，这个结果是符合预期的。
 ![take time](/images/Performance/3.png "profile")
- 第五章：不同条件的horovod实验，但超参数都是固定400 epochs，20 batch size， 0.001 learning rate，改变的是集群节点数、memory mode，并且关注三个东西：time，loss，accuracy
 只是对epochs进行划分，**由于我们没有对训练数据进行切分，所以每个epoch都对数据进行一次完整的遍历**。依据集群最多节点时将一个节点一次epoch耗费时间当成总epoch时间，节点越少，计算时间就得多依赖几个epoch，简单说就是**所有实验处理相同数据量的时间对比，集群最大的关注一个epoch，剩下的可能需要n个epoch才能处理相同数据量**
 ![从结果看，准确率需要多个epoch，但处理的数据量是一致的，待解释](/images/Performance/4.png "horovod result")
 400节点是1个epoch，100节点是4个epoch，25节点是16个epoch（保证了处理400份数据），数据量是一样的，节点多的并行处理，所以速度更快。但loss，acc都不如节点少的。
 结论是：适当的增加学习率可以提高准确率，模型训练需要适当的epoch数来达到高准确率
 （对timeline的使用让我怀疑人生，怎么分析界面都是不一样的，看着头疼）

**相关文献**
> NT3[代码](https://github.com/ECP-CANDLE/Benchmarks)
> [NVProf](https://docs.nvidia.com/cuda/profiler-users-guide/index.html)
> [Python Profilers](https://docs.python.org/2/library/profile.html)

## The third pass

**实验**

**重现**

**体会**
> 作为一篇系统测试论文，整个流程介绍的比较仔细，并且开源实验代码，只是环境可能比较特定，但也容易让人进行重现，对timline的分析，cProfile工具使用，horovod框架选择原因等都有比较细致说明。其实还是讲得比较简单，参考意义也不大:-)
