---
title: ZeRO Memory Optimization Towards Training A Trillion Parameter Models
date: 2020-02-22
tags: install
category: [deeplearning,distributedTraining]
---
2020-02-24梳理
# ZeRO & DeepSpeed
## 1 ZeRO
2019年10月微软的几位大佬Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He发表了一篇论文《ZeRO: Memory Optimization Towards Training A Trillion Parameter Models》。主要针对当前模型越大越难训练的困境，主流的模型并行基本都依托单机多卡实现，训练限制比较大；但如果扩展到多机多卡，通信与规模效率都不高，所以作者通过结合数据并行和模型并行的优势来达到对内存优化，并实现了训练超大模型的可能。
`ps：把该文的内存等同于显存，就看使用CPU还是GPU训练。`
### 1.1介绍
深度学习分布式训练的两个基石：应对大数据量的数据并行和应对大参数量的模型并行。两种方式各有利弊：
	1. 数据并行：不能解决每个设备内存占用问题，基本上单机训练不了15亿参数的模型；但其容易使用，可大规模部署，计算与通信效率高。
	2. 模型并行：由于细粒度计算和昂贵的通信，模型并行性很难有效地扩展到单个节点的多个设备之外，更不用说多个机上；但其可以降低单个设备的内存/显存占用。
作者结合数据并行和模型并行的优势来达到内存优化，与跨数据并行进程复制内存状态的基本数据并行性不同，ZeRO对模型状态(也就是训练时模型在显存中的占用部分，包括参数、梯度等)进行分区存储，从而使模型大小与设备数量成线性关系。此外，它通过计算和通信重调度(也就是将一些可重新计算的变量不进行保存，以此降低内存占用)以及降低运行大型模型所需的模型并行度来保持扩展效率。
### 1.2模型并行
近年比较常见的模型并行方案基本分两类：pipeline parallelism和Tensor Slicing，其实可以认为是纵向和横向划分网络的方式。
	1. pipeline parallelism：需要按层划分，并在数据集上做优化，以此达到一部分数据并行，常见开源框架如GPipe和PipeDream。
	2. Tensor Slicing：对Tensor进行切分，是拆分层的一种方式，如Megatron-LM和Mesh-TensorFlow。Nvidia开发的Megatron-LM可以在单个DGX2节点中16张卡上训练200亿参数的模型，并支持多机模型并行，当前只是针对NLP任务。
作者不是通过扩展模型并行的规模来训练大模型，而是通过分析训练时内存在保存的是什么，通过对保存的内存进行优化来达到训练大模型的可能性。
### 1.3内存占用分析
15亿参数的GPT-2模型，以16位浮点型统计参数，总共3GB，但用pytorch或TensorFlow实现的，却不能在单个32GB的GPU上训练，这是为什么，还有哪些占用了GPU的内存。在模型训练时主要内存消耗在以下几个方面：
	1. activations`[1]`，激活函数部分，待深入了解。
	2. OGP states，由优化器状态、参数梯度和参数本身组成的张量。
	3. temporary buffers，大模型需要GPU间做gradient all-reduce操作，会有融合缓冲张量产生，也会占一部分内存。
这些都是缓存在GPU显存中的，针对activations，如果不缓存，重新计算代价又如何呢，以Megatron-LM实现为例，33%的重新计算开销为代价，可以轻松地消除激活所需的几乎所有内存。基于Megatron-LM已实现，可以重用，作者在本文主要关注后面两个：OGP、temporary buffers。
OGP：以Adam优化器为例，主要存储两部分优化器状态
	1. 平均动量的时间，
	2. 梯度的方差来计算更新，
所以用Adam优化器来训练模型，必须保证足够的内存来存储动量和梯度方差的副本，此外还需要内存来存储梯度和权重。
#### 1.3.1模型状态
模型在内存中的状态主要分三部分：1、优化器状态，如momentum和Adam的方差；2、梯度；3、模型参数。简称模型状态的OGP，O对应优化器，G对应梯度，P对应参数。
作者考虑结合数据并行和模型并行的优势，通过数据并行来将这些状态划分到不同设备，以此来降低单个设备的内存占用。即通过跨数据并行进程(而不是复制它们)划分OGP模型状态来消除数据并行进程之间的内存冗余，它通过在训练期间使用动态通信调度来保持数据并行性的计算粒度和通信量，从而保持计算/通信效率。
如下图中以Adam优化时，内存消耗分成三部分，参数，梯度，优化器（优化器又分：参数、动量、方差，为什么优化器还细分，需以Adam为例），以混合精度计算：(2+2+K)*，第一部分为参数，以FP16存储，共 2Byte；第二部分为梯度，以FP16存储，共 2Byte，第三部分为优化器，以FP32存储，共Byte。（该种计算方式与之前的有些出入，待分析）
 ![](/images/ZeRO/1.png "")
如上图，是模型参数总量，K表示优化器状态的内存量，不同优化器，需要量不同，上图中以Adam优化器为例，一个优化器由参数、动量、方差组成。Nd表示数据并行的节点数。当不使用数据并行时，存储的内容还是与正常模型训练一样，只是利用数据并行来降低内存占用。
 ![](/images/ZeRO/2.png "")
基于其理论分析，如上图，在32GB的V100 GPU上，单卡时，75亿的模型需要占用120GB显存，但利用ZeRO后，在划分模型的OGP状态后，4卡时每个设备只需要保存30GB，1024卡时每个设备只需保存0.12GB，有极大的内存优化收益。理论上提升还是很大，实际呢，下图是实际测试结果。
 ![](/images/ZeRO/3.png "")
如上图，设备为32GB的V100 GPU，第一列是模型并行使用卡数，第二列是总的卡数，与模型并行卡数相除，可以知道数据并行都是64个节点。中间一列是理论可容纳参数量，右边一列是实际测试容量。
Pos表示只划分优化器状态时，以单卡来说，理论只可容纳20亿参数的模型，使用ZeRO的Pos划分后可容纳76亿参数量的模型；实际只能容纳13亿参数，使用ZeRO的Pos划分后可容纳62亿参数。说明内存占用是降下来了，可容纳更大参数量的模型。
因为论文只实现了对优化器状态的划分，所以只比了该部分。
#### 1.3.2Temporary Buffers
对于大型模型，用于存储中间结果的临时缓冲区会消耗大量的内存，诸如梯度的all-reduce计算之类的操作，为了提高该操作的处理速度，往往会将所有梯度融合到一个单一的扁平缓冲区中再进行操作，这部分会占一部分内存。
### 1.4优化技术
作者明说是结合数据并行与模型并行来达到此优化手段，使用Pytorch框架，Pytorch数据并行方式是Ring All-Reduce，也就是借助ring这个顺序进行划分模型各状态。
#### 1.4.1Pos : Optimizer State Partitioning
partitioning：利用数据并行，如果使用Nd节点，那么将Optimizer states等分成Nd份，每个进程只保留并更新1/Nd份Optimizer states，之后在每个训练步骤之后通过all-gather来得到所有进程的参数更新，也就是数据并行越多，保留的越少。
#### 1.4.2Pg: Gradient Partitioning
由于每个数据并行进程只更新其对应的参数分区，因此只需要减少对应参数的梯度，因此，由于每一层的每一个梯度在反向传播时都是可用的，我们只在负责更新相应参数的数据并行过程中减少它们，减少后，我们不再需要梯度和他们的内存可以释放（但计算时需要把）
#### 1.4.3Pp: Parameter Partitioning
正如优化器状态和梯度一样，每个进程只存储与其分区对应的参数。当分区外部的参数需要进行正向和反向传播时，它们通过广播从相关的数据并行进程接收。乍一看，这可能会导致大量的通信开销，但我们证明，这种方法只会将基线数据并行系统的总通信容量增加到1.5倍，同时使内存减少与Nd成比例
### 1.5实验
作者只实现了优化器的分区，所以只跟2019年NVIDIA发布的Megatron-LM框架比对，两者比的是规模容量与处理速度。因为ZeRO是以Pytorch实现，所以框架限制比较大。
#### 1.5.1吞吐量
使用DGX2节点，每节点16个32GB的V100 GPU，比对现有Megatron(NVIDIA开源的大模型训练库，可支持多机多卡分布式)分布式框架，对模型参数大小为8B，20B，40B，60B，80B和100B(一个B就是10亿)的模型进行训练，采用模型并行加数据并行的混合模式。
 ![](/images/ZeRO/4.png "")
从上图可知，在相同配置下，ZeRO单个GPU处理能力优于Megatron，因为能容纳更多batch size的数据，而且单机模型并行容纳的模型规模更大。
#### 1.5.2GPT-2训练
15亿参数的GPT-2模型，以16位浮点型(即2Byte)统计参数，使用Adam优化，参数量总共需(2+2+6)*15亿 Byte=15GB内存。单单模型的OGP就需要15GB，单个16GB的GPU不能训练。
 ![](/images/ZeRO/5.png "")
都是使用相同数量GPU，ZeRO可以不需要进行模型并行就能训练，而Megatron需要进行模型并行才可训练，模型并行本身就极大消耗通信，所以Megatron每秒处理样本数不如ZeRO，大约慢了3.75倍。
#### 1.5.3总结
论文只是提供了一种训练大模型的思路，大规模模型并行通信效率不高，那就从最基本的降低内存使用上进行优化。而且作者也实现了其中优化器的划分，训练效率上是完胜前段时间NVIDIA开源的Megatron，但论文仅仅是对训练速度上的实验，未对训练收敛、准确性等问题进行验证。
 ![](/images/ZeRO/6.png "")
此外，如上图，论文实验都是使用高配置，实际操作可能达不到论文结果。
# 2 DeepSpeed
DeepSpeed`[2]`因为开源时间不长，也未进行实际验证，只列出如下几点：
	1. DeepSpeed是微软于2020年2月开源的一个深度学习分布式训练库，是2019年10月微软发表的论文《ZeRO: Memory Optimization Towards Training A Trillion Parameter Models》的实现，与pytorch兼容，开源时间不长，文档不全，也不成熟，社区不活跃。
	2. DeepSpeed目标是让模型更容易训练，有点类似horovod，需要修改代码适配来跑数据并行和模型并行（代码部分需确定调研才深入），但没介绍什么方式进行数据并行和模型并行。本身DeepSpeed就是再介绍如何结合数据并行和模型并行的优势，来降低内存使用，所以模型并行还是得依赖pytorch的实现。
	3. 但其模型并行并未有很大介绍，完全是在介绍如何优化模型，使GPU在训练时占用更少的显存，并结合数据并行来达到训练规模扩大和内存利用率的提高。号称是能单机模型并行训练一万亿参数的模型，如果单机能处理这么大的模型，真不需要进行分布式并行了。
	4. DeepSpeed的每个技术都是分开的，单独提供了ZeRO优化技术，可以与其他技术结合使用，如 PipeDream，GPipe，Megatron-LM。
# 3 总结
微软该项技术只是结合数据并行和模型并行的一种优化手段，未实现真正意义上的模型并行，如果需要跨机器的多机多卡模型并行，该方案不是可选，但未来如果有对应分布式模型并行，是否可以结合，降低内存消耗呢。
ZeRO&DeepSpeed从另外一个角度来看待大模型的训练，直接降低内存占用，通过合理的划分模型在内存中的状态。按作者分析是依据数据并行节点数等分模型各个状态，是否需要层与层之间的联系。此外DeepSpeed由Pytorch实现，框架限制比较大。
总之DeepSpeed才刚刚开源，还有许多未知的，有待验证。而且ZeRO论文是与NLP任务的Megatron对比，跑的也是GPT-2模型，不知道是否支持CV领域相关的，未知信息较多。
现在模型并行为什么分布式难实现，个人理解：
	1. 网络需要划分到不同进程中，真正的分布式实现，但现有方式都是定制化实现，网络划分如何做到自动化呢；Megatron-LM可能是这类，未调研。
	2. 如果能将集群GPU统一以一个进程识别，该方式是最理想的，这样无论哪个框架其实都可容易实现分布式模型并行，但如何统一识别，GPU云化？扩大服务器卡槽数？好像有做这方面的[rCUDA](http://www.rcuda.net/index.php/what-s-rcuda.html)，但不是开源的
3. 已有的对模型进行压缩，优化，使用半精度保存等等手段，当模型参数巨大时，不是有效解决办法，有其他方式降低模型内存占用吗？GPipe、DeepSpeed都是这类。
老铁，还是买更大容量的GPU吧。

Reference
	[1]`Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. Training deep nets with sublinear memory cost. CoRR, abs/1604.06174, 2016.`
	[2]`https://github.com/microsoft/DeepSpeed`

======================================================================================================================================================================================

2019年10月由微软的几位大佬Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He发表的一篇论文。
主要针对当前模型越来越大的困境，通过内存优化得技术来达到训练超大模型的可能。
（今天日子真是，20200222）
# Three Pass

## The First Pass
**列出**
1. 标题，摘要，介绍
	- 标题：ZeRO: Memory Optimization Towards Training A Trillion Parameter Models；标题比较直白，通过内存优化技术，来训练1万亿参数的模型，定名是超大模型的训练，现有技术要么特定网络特殊处理，要么使用现有模型并行计算，如mxnet、torch、tf的模型并行，但现有模型并行基本都是单机多卡模型并行，如此大参数量的模型，可能单机存放不下，那如何分布式呢，论文为什么不用分布式模型并行，而仅仅是内存优化就解决了呢。
	- 摘要：训练10亿到万亿参数的模型是很困难的，也是当前DL发展面临的一个困境，现有的解决方案在同时获得内存和规模(计算/通信)效率方面存在限制。
	1、数据并行：不能解决每个设备内存占用问题，基本上单机训练不了15亿参数的模型
	2、模型并行：由于细粒度计算和昂贵的通信，模型并行性很难有效地扩展到单个节点的多个设备之外，更不用说多个机上
	作者另寻僻径，通过优化内存来达到内存和规模的效率提升，**与跨数据并行进程复制内存状态的基本数据并行性不同，它对模型状态进行分区，从而使模型大小与设备数量成线性关系。此外，它通过计算和通信重调度以及降低运行大型模型所需的模型并行度来保持扩展效率。**具体如下，继续更下去。
	- 介绍：相关背景与面临挑战，这其实也是当前DL的一个经验-提高模型的大小能获得更好的准确率，并且介绍NLP中的大模型，Bert-large总共0.3B(3亿参数)，GPT-2总共1.5B(15亿参数)，看来还是NLP模型比较大:-)。
	同时介绍当前并行训练的两种方式：数据并行和模型并行，其中数据并行虽然能很容易做到规模化，但每个设备加载模型的内存都是一样的，不能帮助我们训练大模型。所以在一个模型不能加载到单个设备上时，就得使用模型并行来进行处理了，现存比较常用的方法有两类：1、pipeline parallelism，如GPipe和PipeDream；2、Tensor Slicing，如Megatron和Mesh TensorFlow，类似Megatron可以在单个DGX2节点中16张卡上训练200亿参数的模型，也是单机多卡模型并行。
	模型并行面临的困难，即使Megatron可以单机可以处理200亿参数的模型，如果处理1万亿参数量，需要50台机子，总共800张卡来进行模型并行，虽然做到了多机多卡模型并行，但需要付出极大的通信消耗，性能也未必可以提升。**就是即使将模型并行分布式化，也不能有效训练，那是不是可以换一个思路，使单机多卡模型并行就可以训练呢，**这就是作者接下来的想法。
	**看点一**，分析训练时，内存状态：主要内存占用有三部分，1、优化器状态，如momentum和Adam的方差；2、梯度；3、模型参数。简称模型状态的OGP
	**看点二**，分析模型并行与数据并行特点：1、数据并行容易规模化，计算/通信效率高，但内存利用率低(此处的利用率是每个设备都是相同模型的副本，使用的都是完全一样的内存占用)；2、模型并行容易获得高内存效率，但通常会导致过于细粒度的计算和昂贵的通信，从而降低了规模化效率。**这两种方法都静态地维护整个训练过程中所需的所有模型状态，即使不是所有的模型状态在训练期间都是必需的**
	作者提出的方法：Zero Redundacy Optimizer，**通过跨数据并行进程(而不是复制它们)划分OGP模型状态来消除数据并行进程之间的内存冗余，它通过在训练期间使用动态通信调度来保持数据并行性的计算粒度和通信量，从而保持计算/通信效率。**作者把这种方式叫做ZeRO-powered data parallelism。
	总结起来就是ZeRO可以减少单个设备的内存占用，以此来达到模型并行的效果，并且作者在单机16卡做模型并行，多机64节点做数据并行，总共1024个GPU训练1万亿参数的模型，即有效率又有规模。
	**ZeRO三个主要优化阶段**：使用Adam训练，在结合数据并行或混合并行，作者得出结论-对以下三个状态进行分区能节省4,8,Nd倍的内存，1、optimizer states，2、gradients，3、parameters
	验证1：1、对Optimizer的优化，没用模型并行，单个32GB的V100 GPU可以处理60亿参数，而单单用pytorch只能处理15亿参数，有4倍提升；2、结合单机模型并行，ZeRO可以处理1000亿参数的模型（到论文这阶段，只实现了对Optimizer的优化，剩下两个还未做）

	这部分介绍挺实在的，不能从一个角度突破，就换另一个思路。
2. 章节
	- 1 Extended Introduction
	- 2 Background: Data Parallel Training, Model Parallel Training
	- 3 Where Did All the Memory Go?: Optimizer States, Gradients and Parameters, Temporary Buffers
	- 4 ZeRO: Insights and Overview
	- 5 ZeRO: Memory Optimization: Optimizer State Partitioning, Gradient Partitioning, Parameter Partitioning, Constant Size Buffers, Summary of Memory Optimization
	- 6 ZeRO: Communication Analysis: Data Parallel Communication Volume, ZeRO Communication Volume, Communication Latency
	- 7 ZeRO & Model Parallelism
	- 8 Step Towards 1 Trillion Parameters
	- 9 Implementation and Evaluation of ZeRO-OS: ZeRO-OS Implementation, Evaluation Methodology, Scaling to 6B Parameters with Data-Parallelism Alone, Scaling to 80B-100B Parameters, Up to 6x System Throughput Improvement, Up to 3x Resource Savings
	- 10 Concluding Remarks
	- References

3. 数学原理
	- 如何计算模型参数量
	- 时间提升涉及哪些部分
4. 结论
	- ZeRO算是一种比较优化得技术，作者还能实现用于实验，也说明他们比较深入该领域，但到工业界使用可能还不成熟，毕竟代码开源不久，而且只实现了Optimizer的一种优化方法，如果继续深入，更成熟点，用在工业界也未尝不可。作者也提到了，该技术还处于初步发展阶段，有待改进与认识，最终目标都是使不可能训练的模型变成可能。

5. 文献
	- 大型模型，BERT
	- 相关研究，模型并行框架：GPipe、PipeDream、Mesh-TensorFlow（该文献重复写了，文献5、7是一样的，不知道是不是作者漏了一个文献）、Megatron-lm

**回答**
1. 论文类型
	- 系统原型描述与测试，主要是对新提出的一种内存优化技术ZeRO的功能、实现描述，并实验验证理论结果。

2. 相关内容
	- 模型并行为什么这么难，通信/计算 效率，大规模部署，真正实现应该都具备条件
	
3. 正确性
	- 在单机上尽可能训练超大规模的模型，以现阶段，如果真能在单机16卡，每卡32GB的机器上训练1万亿参数的模型，就真不需要跨机器模型并行了

4. 贡献点
	- 内存优化技术ZeRO，针对Optimizer、Gradient、Parameter进行优化的手段

5. 文献情况
	已读： 
	[1] Gpipe: Efficient training of giant neural networks using pipeline parallelism. ArXiv, abs/1811.06965, 2018. 还是正常的模型并行手段，只是对mini-batch进行再划分，使训练时能减少GPU间的等待时间，提升一些计算并行度

	未读：
	[1] Pipedream: Fast and efficient pipeline parallel DNN training. CoRR, abs/1806.03377, 2018.
	[2] Mesh-tensorflow: Deep learning for supercomputers. CoRR, abs/1811.02084, 2018
	[3] Megatron-lm: Training multi-billion parameter language models using model parallelism, 2019
	[4] An empirical model of large-batch training. CoRR, abs/1812.06162, 2018.
	[5] Training deep nets with sublinear memory cost. CoRR, abs/1604.06174, 2016
	
6. 是否继续
	- 继续，这块是一个很大的坑，如果ZeRO技术成熟应用，单机虽能满足大模型训练，但结合到业务使用，还有很长一段路

## The Second Pass
**图表**
	- 第一章：扩展介绍
	- 第二章：背景
	DL训练的组成，主要在一个step中处理样本，而一个step包含前向和反向
	Data Parallel Training：模型参数是复制到每个GPU的，唯一不同的是mini-batch的数据根据数据并行进程进行拆分，每个进程处理不同的数据，在方向传播是，所有进程的梯度都会求平均，之后再给每个进程处理，也就是OGP状态每个进程都有一份。
	Model Parallel Training：主要是减少每个GPU的内存占用，将模型分割为多个部分，并结合多个设备一同执行前向和反向计算，也就是把计算分发到不同设备进行，这个设备进行加，另外一个设备进行减。模型并行主要有两种划分方式-水平划分，垂直划分
	- 第三章：分析内存占用在哪块，这部分很主要，也是作者的着力点
	分析：15亿参数的GPT-2，如何以16位浮点型统计参数，总共3GB，但用pytorch或TensorFlow实现的，却不能在单个32GB的GPU上训练，这是为什么，还有哪些占用了GPU的内存。在模型训练时主要内存消耗在以下几个方面-1、activations，2、OGP states，由优化器状态、参数梯度和参数本身组成的张量，3、temporary buffers；这些都是缓存在GPU显存中的，针对activations，如果不缓存，重新计算代价又如何呢，以**33%**的重新计算开销为代价，可以轻松地消除激活所需的几乎所有内存，这是Megatron已经实现的，论文作者主要关注后面两个-OGP、temporary buffers
	**OGP**：以Adam优化器为例，主要存储两部分优化器状态-1、平均动量的时间，2、梯度的方差来计算更新，所以用Adam优化器来训练模型，必须保证足够的内存来存储**动量**和**梯度方差**的副本，此外还需要内存来存储**梯度**和**权重**，怎么计算的没咋看懂
	**Temporary Buffers**：大模型需要GPU间做gradient all-reduce操作，会有融合缓冲张量产生，也会占一部分内存，具体多少，如何计算
	- 第四章：ZeRO介绍，看看作者究竟是怎么来减少OGP和temporary buffers的
	**如何在不牺牲效率的情况下减少内存占用?**就说明不是简单的将参数放到CPU或提高模型并行的规模，作者主要依据3个关键点：
	1、数据并行有比较好的规模效益，数据并行比模型并行在计算效率和通信上更占优势，毕竟是可能多个GPU并行计算
	2、数据并行内存效率较低，每个GPU都有所有的模型状态，可能模型状态不是必须的，模型并行就比较好利用内存
	3、无论数据并行还是模型并行，所有模型的状态在整个训练过程中都是保存着的，有些只需要在特定时刻使用即可，作者提到每层的参数只在前向和反向有用，这块没搞懂，参数如果不保存在内存中，有其他方式进行重新计算出来吗，但有些状态确实是可以重新计算，不用保存，只是计算和保存哪个收益更高
	- 第五章：内存优化，如何优化，如何计算各部分占用情况
	**通过对Optimizer states、gradients、parameters进行分区来优化内存**
	还是利用数据并行来进行减少单个GPU占用，如果我只有一个GPU，可能就不能使用这种技术了，有点类似对数据并行的优化。
	【1】Optimizer state partitioning：对于数据并行，如果使用Nd节点，那么将Optimizer states分成Nd份，每个进程只保留并更新1/Nd份Optimizer states，之后再每个训练步骤之后通过all-gather来得到所有进程的参数更新，也就是数据并行越多，保留的越少
	【2】Gradient Partitioning：
	【3】Parameter Partitioning：
	
	- 第六章：通信问题
	
	- 第七章：模型并行什么时候用-当单独使用数据并行的聚合批处理大小太大而无法获得良好的收敛性时

**相关文献**
## The Third Pass
**实验**

**重现**
