---
title: One weird trick for parallelizing convolutional neural networks
date: 2020-02-09 20:44:24
tags: introduce
category: [deeplearning,distributedTraining]
---
该论文是2019年6月25日google发表的一篇文章，作者为Alex Krizhevsky大牛

# Three pass
明确研究背景，
存在什么问题，
基于什么**假设前提**下，使用什么方法，
实验结果如何，得出什么结论
## first pass
**列出**
1. 标题，摘要，介绍（背景，方法，结果）
标题：One weird trick for parallelizing convolutional neural networks，卷积神经网络分布式计算
摘要：作者提出一种多GPU训练卷积神经网络的方法，效果比现有方法都好
介绍：作者已经SGD来训练卷积网络，并提出两种算法：同步SGD，非SGD算法，只是说如何训
2. 章节
> Existing approaches
> Some observations
> The proposed algorithm:Weight synchronization, Variable batch size
> Experiments:Results
> Comparisons to other work on parallel convolutional neural network training
> Other work on parallel neural network training
> Conclusion
> References
3. 数学原理
什么数学原理呢，讲的类似对分布式SGD的一些改变，已达到相同或更好的效果，与输入batch size等信息相联系
基本还是SGD的数学原理改进
4. 结论
不需要适配多GPU，只是提到batch增大时的影响，对其如何构建卷积网络的分布式还是有看头的
5. 文献
相关：
未阅读：
已阅读：

**回答**
1. 论文类型：技术实验论文
2. 上下文关联：分布式，多GPU，batch size增大对收敛影响
3. 正确性：多GPU同步训练就是增大了batch size，对SGD有何影响进行理论和实验说明
4. 贡献点：可以帮助理解SGD如何同步，有哪些因素影响在分布式下
5. 条理：清晰，介绍了现存的分布式方法
6. 是否继续：可以继续

## second pass
（卷积网络：卷积部分计算量大(90~95%)，但参数少(5%)；全连接部分计算量少(5~10%)，但参数多(95%)）
这是作者观察的结果，所以针对卷积网络不同部分进行不同类型的并行，卷积部分用数据并行，全连接部分用模型并行，这也是作者主要实验手段
**图表**

**相关文献**

## third pass
**重现**