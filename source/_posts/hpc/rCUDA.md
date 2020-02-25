---
title: rCUDA Reducing the Number of GPU-Based Accelerators in High Performance Clusters
date: 2020-02-25
tags: introduce
category: [deeplearning,distributedTraining,HPC]
---
2011年的一篇论文，有关如何远端机器的GPU技术。

## first
**列出**
1. 标题、摘要、介绍
	- 标题：rCUDA: Reducing the Number of GPU-Based Accelerators in High Performance Clusters，从标题可以看出论文是对CUDA进行了改进，减少HPC中的GPU数量？
	- 摘要：讲了一个什么问题-如何减少GPU的使用，为了节能
	- 介绍：作者一直在介绍如何节省能耗，在HPC中一个节点就需要消耗50%能源，如果虚拟化GPU，使每个人都共享GPU资源，而不是每个人单独一份GPU资源，可以达到一定得节能效果。现有虚拟化技术如VM、VB、KVM等都是对整个系统进行虚拟，而非对特定设备如GPU进行虚拟化，所以不合适，作者试图在找一种能在本地使用远端GPU资源的方法，通过本地一个虚拟GPU来与远端物理GPU进行绑定，也就是本文的rCUDA技术。作者的意思有点类似10个人用HPC的话，每个人都必须有一块GPU，如果用rCUDA可以使用远端GPU，那么10个人共用一块即可。（作者初衷是为了减少GPU数量的使用，用在我们这，主要是如何访问远端GPU）
2. 章节
	- PRIOR WORK ON VIRTUALIZATION: Virtualization Taxonomy, GPU Virtualization for Graphics Processing, GPU Virtualization for GPGPU 先前相关工作介绍
	- Background on NVIDIA CUDA：CUDA框架简介
	- CUDA Remoting framework: Architecture, Client Side, Server Side, Asynchronous Memory Transfers CUDA框架详解
	- Discussion
	- Experiments：Usability, Performance
	- Power Saving
	- Conclusions
	- References
3. 数据原理
	- 技术框架实现，无数学部分
4. 结论
	- 一直在说HPC中为每个机器配GPU，在能源消耗上是巨大的，作者想的就是有个中心服务器安装GPU即可，其他人都共享这些GPU资源，并提供rCUDA技术使每个人能像使用本地GPU一样来使用远端的这些GPU资源。
	- 虽然rCUDA在实现上依赖NVIDIA的CUDA，有些兼容性不好，避免使用CUDA C即可。
	- 此外因为使用的是远端的GPU资源，计算时间可能消耗较大
5. 文献
	- 文献时间都比较早，但这个方向还是大有人在做
	- vCUDA: GPU accelerated high performance computing in virtual machines
	- An efficient implementation of GPU virtualization in high performance clusters
	- Modeling the CUDA remoting virtualization behaviour in high performance networks

**回答**
1. 论文类型
	- 技术类文章，对GPU资源虚拟化与远端GPU访问技术
2. 相关内容
	- 对特定设备的虚拟化技术
3. 正确性
	- 结合现有gPRC技术，该技术是可行的，就看性能如何
4. 贡献点
	- 提供remote GPU访问技术，好像未开源
5. 相关文献
	- 
6. 是否继续
	- 可以继续了解实现原理
## second
**图表**
	- 第一章
	- 第二章：相关技术介绍
	作者将现有虚拟化技术划分为两类：
	1、前端虚拟化（面向应用级别）：前端技术有可分为两类
		> 1. **API remoting**：API调用被拦截并转发到远程主机上执行，也叫API拦截。作者以该方式进行实现。
		> 2. device emulation：设备模拟开销大
	2、后端虚拟化（面向硬件级别）：设备驱动在客户端，直接访问物理设备，这是面向VM时的技术，也就是该设备与客户端必须在同一个机器上。这种技术不是作者使用的，因为设备在远端机器上。
	此外API remoting方式实现的GPU虚拟化，依赖OpenGL或Direct3D来进行图形渲染，有些许问题，作者使用CUDA，CUDA主要与通用计算相关，不考虑图形表示问题
	- 第三章：CUDA简介
	GPU：hig-speed DRAM，PCI-Express bus（PCIe）连接计算机
	CUDA允许使用NVIDIA gpu作为协处理器，以加速程序的某些部分，通常是那些每个数据都有高计算负载的部分。
	CUDA提供两种C语言的API：
	1、the low-level Driver API
	2、the high-level Runtime API
	- 第四章：rCUDA框架介绍
	1、如下rCUDA的框架图，一个server端进行管理GPU资源，client端使用封装的CUDA API通过网络来进行访问，原理简单，实现技术是否有难点呢
	![](/images/rCUDA/1.png "rCUDA")
	
	- 第五章
