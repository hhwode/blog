---
title: 贪心学院NLP公开课
date: 2020-02-13 20:44:24
tags: introduce
category: deeplearning
---
# #2020-02-13
# 介绍
常用技术：
信息抽取、知识图谱、对话系统、智能问答 热门方向，这几年

# 词向量与ELMo模型
 研究词的表示方法，词向量有两个阶段：
 1. 第一阶段：CBOW，SkipGram，一个单词对应一个词向量
 2. 第二阶段：ELMo，GPT，Bert，XLNet，一个单词对应多个词向量
 【贪心学院的NLP技术分享，直播因为人员在4000多人崩溃了，等待通知】

【2020-02-15 B站直播李文哲老师https://live.bilibili.com/11869202】
# 基础回顾（词向量，语言模型）
深度学习使用更少的参数来表示
1. 独热编码能否表示单词相似度

2. 语言模型：对于句子的计算
相关知识点：
	1. Chain Rule，Markov Assumption
	2. Unigram，Bigram，Ngram
	3. Add-one smoothing，Good-turning Smoothing
	4. Perplexity

3. 基于分布式表示模型总览	
  ![]( '分布式表示模型总览')
	1. Global vs Local
	 矩阵分解是global方法，从全局进行，优势：从全局考虑，信息量多，劣势：计算量大
	 SkipGram是local方法，每次训练基于窗口式，优势：随时添加数据，劣势：无全局信息
	2. Local又分基于LM vs 非LM
	 SkipGram是非LM
	 LM就是可写出目标函数的，因为有上下联系

4. 学习路径
  ![]( "学习路径")

# NLP核心：学习不同语境下的语义表示
基于LM的方法词有前后顺序依赖，一般用2nd order Markov Assumption
1. 基于非语言模型的方法：SkipGram、CBOW
	1. SkipGram：

2. 基于语言模型的方法：NNLM

以上两类方法训练的词向量只是固定的词向量，一个词对应一个词向量

解决问题：对于一个单词依据上下文有不同的词向量
一种解决方案：ELMo
核心思想：
	- 基于语言模型的训练：使用LSTM作为基石
	- 启发来自深度学习中的层次表示(Hierarchical Representation)

# 基于LSTM的词向量学习
单向LSTM=NNLM
双向LSTM=SkipGram/CBOW

# 深度学习中的层次表示以及Deep BI-LSTM
层次表示：
就是网络层次结构，一层学一层特征，单层没有含义，但所有层合起来就表示这个样本

# ELMo模型
为什么ELMo要使用Deep BI-LSTM

# 总结

# #2020-02-17

# Attention & Transformer

## 

# #2020-02-19

# BERT模型

Polysemous words：一词多义

LSTM：训练是迭代的，串行的
Transformer：训练时并行的，使用位置嵌入来帮助

Transformer核心组成：
1. self-Attention
2. Layer Norm
3. Skip Connection
4. Forward Network
5. Positional Encoding

BERT: Bidirectional Encoder Representation from Transformers
pre-training model + fine-tuning

BERT输入、组成，都是有其理由的
由完形填空启发，




