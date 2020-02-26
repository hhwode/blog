---
title: On the Design of a Demo for Exhibiting rCUDA
date: 2020-02-26
tags: introduce
category: [deeplearning,distributedTraining]
---

2015 15th IEEE/ACM International Symposium on Cluster, Cloud and Grid Computing，对rCUDA使用介绍

## first
**列出**
1. 标题、摘要、介绍
	- 标题：On the Design of a Demo for Exhibiting rCUDA，rCUDA的使用例子
	- 摘要：CUDA是NVIDIA开发的一个并行计算平台，主要是降低程序运行时间，rCUDA是一个中间件，提供远程访问CUDA兼容设备的能力，而不需要本地节点具有该设备。本文主要是介绍一个例子。
	- 介绍：现在困境是GPU数量多（成本高），使用率低，所以有rCUDA共享GPU资源，它以一种应用程序不知道访问远程设备的方式授予应用程序对安装在集群其他节点上的gpu的并发访问权（即本地访问GPU，但访问的是远程GPU资源），并且不需要修改应用程序代码，开销也只是一小部分
2. 章节
	- rCUDA: Remote CUDA
	- Applications used in the demo: Color Image to Grayscale Conversion, Image Blurring
	- rCUDA demo description: Equipment used, Description of the Demo, Performance Results
	- Acknowledgment
	- References
3. 数学原理
4. 结论
5. 文献

**回答**
1. 论文类型
	- 测试系统类型
2. 相关内容
	- rCUDA框架，适用范围
3. 正确性
	- 
4. 贡献点
	- 提供一个使用例子，测试结论证明可行
5. 相关文献
	- NVIDIA CUDA C Programming Guide 6.5, 2014
	- A complete and efficient cuda-sharing solution for HPC clusters,” Parallel Computing (PARCO), vol. 40, no. 10, pp. 574–588, 2014.
	- Influence of infiniband FDR on the performance of ˜ remote GPU virtualization,” in IEEE International Conference on Cluster Computing (CLUSTER), 2013, pp. 1–8.
	- Intro to Parallel Programming, 2015
6. 是否继续

## second
**图表**
1. 每章分析
	- 第一章：
	- 第二章：rCUDA介绍
		远端有个GPU资源，所有CUDA程序都可以通过网络来使用，类似将代码下发到远端的GPU执行，结构如下图所示
	![](/images/rCUDA/11.png "rCUDA")
		rCUDA中间件拦截应用程序对CUDA API的调用，并将其转发给远程GPU
		rCUDA提供两种通信方式：TCP/IP和InfiniBand Verbs API
		本文最新的rCUDA版本是5.0，支持CUDA Runtime和Driver API 6.5，并且支持CUDA对应的库：cuBLAS、cuFFT、cuRAND、cuSPARSE
	- 第三章：示例，介绍基本运算过程
	作者做了两类图像过滤：彩色图到灰度图的转换和图像模糊
	1、在彩色图转灰度图中，因为是对每个像素做同样的转换，所以可并行计算
		1.1、彩色图每个像素4个值表示，RGBA，红绿蓝和透明度，都是一个byte表示，取值范围0~255
		1.2、灰度图每个像素1个值表示，gray，使用一个byte，取值范围0~255
		1.3、转换使用了NTSC公式，暂未深入了解，只需知道转换公式:`I=0.299*R+0.587*G+0.114*B`
	2、图像模糊：模糊图像是在每个像素及其相邻像素上应用一个根据期望的失真程度而变化的滤波器
	- 第四章：实验介绍 
	1、实验设备：两台1027GR-TRF Supermicro servers(超微型计算机服务器)，相同配置
		1.1、处理器：两个Intel Xeon hexa-core processors E5-2680 v2，2.8GHz
		1.2、内存：32GB的DDR3 SDRAM，1600MHz
		1.3、网卡：1 Mellanox Connect-IB (FDR) dual-port InfiniBand adapter. 56Gb/s
		1.4、系统&软件：RedHat Enterprise Linux Server 6.4，Mellanox OFED 2.1-1.0.0，CUDA 6.5 with NVIDIA driver 340.29
		1.5、显卡：NVIDIA Tesla K80，24GB
	2、连接方式如下：示例在节点A中运行，而节点B托管一个rCUDA服务器
	![](/images/rCUDA/12.png "connect")
	3、实验结果显示：结果为10次执行的平均值，观察到的最大相对标准偏差(RSD)为0.077
	![](/images/rCUDA/13.png "result")
	上图是在CUDA的基础上rCUDA的开销，开销有三处：
		1.1、传输：rCUDA通过网络传输数据给CUDA的开销
		1.2、计算：CUDA内核在GPU的计算时间
		1.3、CUDA调用：当使用rCUDA时，对CUDA API的调用转化为小规模的网络传输，这将根据网络延迟增加rCUDA开销，类似rCUDA需要起CUDA
	![](/images/rCUDA/14.png "compute")
	上图是CUDA内核的计算耗时