---
title: Tensorflow Large-Scale Machine Learning on Heterogeneous Distributed Systems
date: 2020-02-07 20:44:24
tags: article
category: [deeplearning,tensorflow]
---
2015年google提出的一篇关于TensorFlow如何分布式训练的论文。

# Three pass

## the first pass
**列出**
1. 标题、摘要、介绍
标题：Tensorflow Large-Scale Machine Learning on Heterogeneous Distributed Systems，TensorFlow做分布式，分布式涉及：多台机器如何通信，梯度如何汇总
摘要：介绍工具（TensorFlow一个接口，用于机器学习算法的实现），存在什么问题，现有方法如何解决，能否解决，本文提出什么解决方法，结果是什么
介绍：主要是一些背景知识，
2. 章节标题
> Programming Model and Basic Concepts: Operations and Kernels, Sessions, Variables
> Implementation: Devices, Tensors, Single-Device Execution, Multi-Device Execution, Node Placement, Cross-Device Communication, Distributed Execution, Fault Tolerance
> Extensions: Gradient Computation, Partial Execution, Device Constraints, Control Flow, Input Operations, Queues, Containers
> Optimizations: Common Subexpression Elimination, Controlling Data Communication and Memory Usage, Asynchronous Kernels, Optimized Libraries for Kernel Implementations, Lossy Compression
> Status and Experience
> Common Programming Idioms: **Data Parallel Training, Model Parallel Training, Concurrent Steps for Model Computation Pipelining**
> Performance
> Tools: Tensorboard: Visualization of graph structures and summary statistics, Performance Tracing
> Future work
> Related Work
3. 数学原理
技术性，无数学部分
4. 结论
描述了一个数据流处理框架TensorFlow，并且开源
5. 相关文献
1、Theano: A CPU and GPU math expression compiler. In Proceedings of the Python for scientific computing conference (SciPy), volume 4, page 3. Austin, TX, 2010.
2、[FlumeJava: easy, effi-cient data-parallel pipelines](http://x86.cs.duke.edu/courses/fall13/cps296.4/838-CloudPapers/FlumeJava.pdf). In ACM Sigplan Notices, volume 45, pages 363–375. ACM, 2010.
3、[cuDNN: Efficient primitives for deep learning](https://arxiv.org/pdf/1410.0759.pdf). arXiv preprint arXiv:1410.0759, 2014.
4、[Project Adam: Building an efficient and scalable deep learning training system](http://www.cs.otago.ac.nz/cosc440/readings/osdi14-paper-chilimbi.pdf). In 11th USENIX Symposium on Operating Systems Design and Implementation (OSDI 14), pages 571–582, 2014.
5、Torch: A modular machine learning software library. Technical report, IDIAP, 2002.
6、[Large scale distributed deep networks](http://ai.stanford.edu/~ang/papers-tofile/large_deep_networks_nips2012.pdf). In NIPS, 2012.
7、[Dryad: distributed data-parallel programs from sequential building blocks](http://its.kpi.ua/itm/lgloba/discipline/%D0%A0%D0%BE%D0%B7%D1%80%D0%BE%D0%B1%D0%BA%D0%B0%20%D1%96%D0%BD%D1%84%D0%BE%D1%80%D0%BC%D0%B0%D1%86%D1%96%D0%B9%D0%BD%D0%B8%D1%85%20%D1%80%D0%B5%D1%81%D1%83%D1%80%D1%81%D1%96%D0%B2%20%D1%82%D0%B0%20%D1%81%D0%B8%D1%81%D1%82%D0%B5%D0%BC/Dryad.pdf). In ACM SIGOPS Operating Systems Review, volume 41, pages 59–72. ACM, 2007.
8、[Caffe: Convolutional architecture for fast feature embedding](https://arxiv.org/pdf/1408.5093.pdf). In Proceedings of the ACM International Conference on Multimedia, pages 675–678. ACM, 2014.
9、Large-scale video classification with convolutional neural networks. In Computer Vision and Pattern Recognition (CVPR), 2014 IEEE Conference on, pages 1725–1732. IEEE, 2014.
10、[One weird trick for parallelizing convolutional neural networks](http://de.arxiv.org/pdf/1404.5997). arXiv preprint arXiv:1404.5997, 2014
11、[Naiad: a timely dataflow system](https://people.eecs.berkeley.edu/~istoica/classes/cs294/15/notes/04-naiad.pdf). In Proceedings of the TwentyFourth ACM Symposium on Operating Systems Principles, pages 439–455. ACM, 2013.
12、[Ciel: a universal execution engine for distributed data-flow computing](https://wwwdb.inf.tu-dresden.de/wp-content/uploads/sose2017-hs5-2.pdf). In Proceedings of the Ninth USENIX Symposium on Networked Systems Design and Implementation, 2011.
13、[Halide: A language and compiler for optimizing parallelism, locality, and recomputation in image processing pipelines](http://people.csail.mit.edu/fredo/tmp/Halide-5min.pdf). ACM SIGPLAN Notices, 48(6):519– 530, 2013.
14、[Hogwild: A lock-free approach to parallelizing stochastic gradient descent](https://arxiv.org/pdf/1106.5730.pdf). In Advances in Neural Information Processing Systems, pages 693–701, 2011.
15、[Chainer: A powerful, flexible and intuitive framework of neural networks](https://chainer.org/)
16、[An introduction to computational networks and the computational network toolkit](http://pdfs.semanticscholar.org/3fc6/8a68225edce1635bc32054c1425db287fba3.pdf). Technical report, Tech. Rep. MSR, Microsoft Research, 2014, 2014.

**回答**
1. 论文类型
技术的描述，探讨
2. 上下文关联

3. 正确性

4. 贡献点
开源深度学习库TensorFlow，避免重复性工作，也算是给定了实现标准
5. 条理
清晰，语句简单，阅读不难
6. 是否继续阅读
TensorFlow开源库的介绍，组成，如何使用，分布式使用，可视化模型结构，追踪模型通信（这就是TensorFlow的timeline）
分布式部分、追踪通信可以阅读，其他不建议，想学TensorFlow直接上官网

## the second pass
**图**
数据并行与模型并行这个概念在TensorFlow这篇论文就提到，不知道最新提出是谁
1. 数据并行：概念比较简单，就是不同的设备计算前向和后向用的数据不同，以达到快速遍历一遍训练样本，通常是将数据分割，每个设备加载自己独有的一份数据
依据不同更新模型方式有同步和异步更新
![](/images/tensorflow-0207/1.png "1")

2. 模型并行：将模型参数拆分，不同设备有模型的不同层参数，依赖模型参数的前后顺序，对设备的利用率不高，因为设备要串行执行前向和反向
![](/images/tensorflow-0207/2.png "2")

3. 利用Linux内核的ftrace和CUPTI等工具统计的日志构成的cpu之间通信情况，这个图用处是用于分析训练耗时与设备使用情况，不清楚能否重现这个或有其他工具替代
![](/images/tensorflow-0207/3.png "3")

4. GPU和CPU的情况，EEG只是说收集和可视化计算图的工具，又与tensorboard不一样
![](/images/tensorflow-0207/4.png "4")

5. 这张图给了一个timeline表示，这也就是最初的TensorFlow timeline
![](/images/tensorflow-0207/5.png "5")

## the third pass
重现
