---
title: Open MPI Goals, Concept, and Design of a Next Generation MPI Implementation
date: 2020-02-11 20:44:24
tags: introduce
category: communication
---

2004年的一篇会议论文，发表于EuroPVM/MPI 2004

# Three Pass

## first pass
**列出**
1. 标题，摘要，介绍
 - 标题：Open MPI: Goals, Concept, and Design of a Next Generation MPI 
 MPI是一种消息传递接口规范，而论文是对这种规范的目标，概念和实现方式的说明
 - 摘要
 MPI的实现很多，但通用性不高，都是针对特定问题的，并且兼容性不好（只能说说吗，后面是否有证据证明），所以论文设计实现了一种兼容性高，扩展性强的OpenMPI
 - 介绍
 背景：MPI规范是针对并行计算框架的通信手段，当然这个规范什么时候提出的(?)，最开始针对单台机器的多进程，但现在基于机器基本构建并行计算框架很流行，且机器不贵，但机器与机器之间配置不同，有可能距离也不同，这就导致很多因素得考虑，因起的进程越多，面临越多问题：
 1、进程控制
 2、资源耗尽
 3、延迟和管理
 4、集群容错性
 5、通信间的优化
 6、网络层传输错误
 ···
 这些背景导致，急需一个东西：它能够提供一个框架来解决新兴网络和架构中的重要问题(为什么，有无其他替代方法)
 OpenMPI可以提供较好的解决方法，论文后面介绍
 
2. 章节
> **The Architecture of Open MPI**
 - Module Lifecycle
> Implementation details
 - Object Oriented Approach
 - Module Discovery and Management
> Performance Results
> Summary
> References

3. 数学原理
> 原型系统介绍，无数学原理

4. 结论
作者提供的OpenMPI实现，支持所有MPI-2规范实现，可提供给个人或企业使用，可提供多个并发用户线程，以及多个处理进程和网络故障的选项，并开源，开发者可贡献
第一个版本是在2004年的 Supercomputing Conference 发布

5. 文献
1、[MPI: A Message Passing Interface Standard](http://www.mpi-forum.org), June 1995. 
2、[MPI-2: Extensions to the Message Passing Interface](http://www.mpi-forum.org), July 1997. 
一个技术的成熟经历时间长，先有理论，再有各式各样的实现，之后统一规划，开源标准版本，不断升级

**回答**
1. 论文类型
 - 原型系统描述

2. 相关内容
 - MPI规范，如何通信
 
3. 正确性
 - 到目前已经是很成熟的技术

4. 贡献点，创新点
 - 基于规范的开源实现，任何人都可以使用

5. 文献情况
 - 文献比较老，都是没有看过

6. 是否继续
 - 可理解MPI框架组成即可，不建议继续

## second pass
**图表**
OpenMPI是围绕MPI Component Architecture(MCA)来设计的，主要包含三部分
1. MCA：
 为所有其他层提供管理服务的主干组件体系结构(服务器)
2. Component framework：
 Open MPI中的每个主要功能区域都有一个相应的后端组件框架，用于管理Modules(客户端?)
3. Modules：
 自包含的软件单元，它导出定义良好的接口，这些接口可以在运行时与其他模块一起部署和组合(配置项?)
 ![](/images/OpenMPI/1.png "OpenMPI框架")
 
**MPI功能描述**如下，后面是基于这些功能形成整个框架的执行流程
> 1. Point-to-point Transport Layer (PTL):
 - 传输网络形式，支持：TCP/IP, shared memory, Quadries elan4, Infiniband, Myrinet
> 2. Point-to-point Management Layer (PML):
 - MPI层和所有可用的PTL模块之间的消息碎片化、调度和重新组装服务。消息处理部分
> 3. Collective Communication (COLL):
 - 
> 4. Process Topology (TOPO):
 -
> 5. Reduction Operations:
 - reduce操作，比如mpi_sum
> 6. Parallel I/O:
 - 并行文件和设备的接入，很多用ROMIO实现，也有适配集群和并行文件系统实现

~~**模块间的交互**~~
~~用COLL模块说明：~~
~~1、MPI_INIT时，COLL找到所有可用的模块，类似加载库~~
~~2、进程必须知道所有的COLL模块才能进行运行~~

**相关文献**

## third pass
**实验**
1. 实验条件
原型系统实现，肯定有对比实验的，性能没别人好，谁会去使用呢
2. 实验结果

**重现**
