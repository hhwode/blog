---
title: DeepSpeed
date: 2020-02-21
tags: introduce
category: [deeplearning,distributedTraining]
---

为什么选择DeepSpeed：distributed training，mixed precision，gradient accumulation，checkpointing
挑战：大规模的模型很难训练在现有模型并行下，DeepSpeed的目标就是让模型更容易训练

刚开源出来的框架

减少训练时模型在GPU的占用，来达到处理更大规模的模型

# 介绍
DeepSpeed
DeepSpeed封装自Pytorch
具备功能：distributed training, mixed precision, gradient accumulation, checkpoints 方便模型部署
只需要修改少部分代码就可以使用DeepSpeed

现有基本应用都是在NLP方面：
1. 对BERT-large训练，
2. 对GPT2模型训练速度比SOAT快3.75倍

内存效率：
1. 使用Zero Redundancy Optimizer(ZeRO)技术，能减少内存使用，减少可高达4倍，比如该GPU最多能容纳8GB参数量的模型，使用ZeRO技术后，该GPU可容纳32GB参数量的模型，有点类似对模型进行了压缩

扩展性：
1. 支持数据并行、模型并行、混合并行

有效快速收敛：

易用性：
1. 只需要修改Pytorch几行代码就可以使用DeepSpeed和ZeRO

# 功能

# 使用
1. 安装：提供了三种安装方式
	1、Azure安装
	2、提供docker镜像，包含所有依赖
	3、本地安装
2. 