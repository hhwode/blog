---
title: Optimization for deep learning theory and algorithms
date: 2020-02-10 20:44:24
tags: introduce
category: deeplearning
---

作者：Ruoyu Sun
2019年12月21日


# Three Pass

## first pass 
**列出**
1. 标题，摘要，介绍
标题：Optimization for deep learning theory and algorithms
摘要：介绍深度学习为什么有效，论文提供深度学习训练优化的理论，讨论1、梯度爆炸与消失，光谱，有效的解决手段初始化和归一化方法；2、回归优化方法，SGD，适配梯度方法和SGD分布式法；3、研究训练存在的问题，bad local minima, mode connectivity, lottery ticket hypothesis and infinite-width analysis.
**可以说是对深度学习一套理论的研究，存在什么问题，怎么解决的**
介绍：
在高层面看，训练有三种条件：Proper neural-net，Training algorithm，Training tricks
并将优化优势大致分为三部分： controlling Lipschitz constants(make convergence possible)，faster convergence， better landscape( better global solutions)
主要还是理论理解，何为neural-net structure
作者还使用分解法来解释，从ML优化分解到DL优化分解
> 监督学习的目标是基于观察样本找到接近真实函数的函数，说直白就是找到一个可替代的解释说明，分三步走：
第一步是找到一个丰富的函数家族(如神经网络)，可以代表理想的函数(表示)
第二步是通过最小化某个损失函数来识别函数的参数(优化)
第三步是使用第二步中找到的函数对不可见的测试数据进行预测，产生的错误称为测试错误(泛化)
**当我们选用一个函数族作为表示时，不去关心该函数是否能将问题优化得很好**
**当进行分析泛化错误时，就认为已经最优化了**
> 分解优化问题，也是可以分三步：
第一步是使算法开始运行，并收敛到一个合理的解，如一个固定点(**convergence**)
第二步是使算法尽快收敛(**convergence speed**)
第三步是确保算法收敛于一个目标值较低的解(如全局最小值)(** global quality**)
2. 章节
> **Problem Formulation**: Relation with Least Squares, Relation with Matrix Factorization
> **Gradient Descent**: Implementation and Basic Analysis: Computation of Gradient: Backpropagation, Basic Convergence Analysis of GD
> Neural-net Specific Tricks: Possible Slow Convergence Due to Explosion/Vanishing, Careful Initialization, Normalization Methods, Changing Neural Architecture, Training Ultra-Deep Neural-nets
> General Algorithms for Training Neural Networks: SGD and learning-rate schedules, Theoretical analysis of SGD, Momentum and accelerated SGD, Adaptive gradient methods: AdaGrad, RMSProp, Adam and more, Large-scale distributed computation, Other Algorithms
> Global Optimization of Neural Networks(GON): Related areas, Empirical exploration of landscape, Mode connectivity, Model compression and lottery ticket hypothesis, Generalization and landscape, Optimization Theory for Deep Neural Networks, Research in Shallow Networks after 2012
> Concluding Remarks
> Acknowledgement
> References:257条引用文献，综述确实难写

3. 数学原理
SGD的数学原理：快速构建一些东西，然后得到一些反馈，根据反馈做出改变，重复此过程。目标是让产品更贴合用，让用户做出反馈，以获得设计开发出的产品与优秀的产品二者之间误差最小
4. 结论
论文回顾了前馈神经网络在训练时相关理论，理解理论的目的有两个：**understanding and design**
> understanding:理解，初始化的作用，过度参数化的认识(待理解：还有很多组件不清楚会对性能有什么影响)
> design:设计， initialization schemes, batch normalization, Adam等使用
5. 相关文献：
已读：
未读：
总共：

**回答**
1. 论文类型：综述性论文，主要介绍深度学习理论，从表示、优化、泛化三方面介绍
2. 论文内容：你读过的其他论文有没有和这个相关的？文章中分析问题用的什么理论基础？
3. 正确性：结论看起来真实有效，针对理解深度学习有帮助
4. 贡献点：综述性的论文，编写难度大，并且资料收集和理解都需要很大精力
5. 条理：条理清晰，从现状、基础到原理分解解释做得足够说明
6. 是否继续：建议继续，能进一步加深理解，为后续无论是具体网络分析还是实验都有帮助，最好打印出来

## second pass
章节：2 problem formulation
监督问题就是在找x, y之间的映射关系，对于分类问题就是整型向量或标量值，对于回归问题就是实值向量或标量值，两者可以认为就是函数嵌套。
**图表**

**相关文献**

## third pass
**重现**

