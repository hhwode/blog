---
title: mxnet
date: 2020-02-13 20:44:24
tags: introduce
category: [deeplearning,framework]
---

# 概念
- **1、mxnet介绍**
 DMLC社区开源的深度学习框架，也是Apache的孵化项目
 特点：
 1. 接口清晰，易用
 2. 显存利用与多卡并行效率高
 3. 部署成熟，PC、移动
 4. 支持多语言python、R、matlab、JavaScript等
 5. 具有静态图编程接口mxnet和动态图编程接口Gluon

- **2、一个模型训练模块：**
 1. 读取数据、数据处理
 2. 定义网络结构
 3. 定义优化器
 4. 定义学习率，固定还是warmup形式
 5. 衡量模型好坏，性能、速度等
 6. 执行网络，从网络获取节点值
 7. 其他能监控网络执行过程的模块，性能分析工具
 
- **3、mxnet核心接口**
 1. Context： 指定运行设备，CPU，GPU
  - mxnet.cpu(0) 类似TensorFlow的 tf.device('/cpu:0')
  - mxnet.gpu(0) 类似TensorFlow的 tf.device('/gpu:0')
 2. NDArray： Python与C++交互的数据对象
  - 升级版Numpy，同时支持CPU和GPU，有ctx输入参数
  - 与numpy转换，nd.asnumpy
 3. DataIter： 为训练提供batch数据
  - 高性能数据读取文件格式：rec，类似tf的tfrecord，caffe的imdb
  - im2rec.py工具创建rec文件
   prefix:lst文件所在目录
   root：图片所在根目录
   --resize：压缩图片使图片最短边resize成指定大小
   --pack-label：如果lst文件中写了多个label，则需要设置为True
   --shuffle：如果要对lst文件进行shuffle后写入rec，则需设置为True
   例子：
  - ImageRecordIter：分类任务图片读取
  - ImageDetRecordIter：检测任务图片读取
  - BucketSentenceIter：用于不定长序列数据的迭代，常用于RNN
 4. Symbol： 定义网络，用于符号式编程接口，只是构建图的结构，没有立即执行
  - 类似隐含定义了权值与偏差
  - 
   Symbol.infer_type:推导当前Symbol依赖的所有Symbol类型
   Symbol.infer_shape:推导当前Symbol依赖的所有Symbol形状
   Symbol.list_arguments:列出当前Symbol所用到的基本参数名称
   Symbol.list_outputs:列出当前Symbol的输出名称
   Symbol.list_auxiliary_states:列出当前Symbol的辅助参数名称
   arguments = 输入数据Symbol + 权值参数Symbol
   auxiliary_states = 辅助Symbol，比如BN中的gamma和beta
   - Symbol如何执行
    需要用bind绑定设备与输入数据，之后计算前向forward
	- Symbol可轻松获取任何一个节点的信息，类似一个链表读取
 5. LR Scheduler：定义学习率衰减策略
 6. Optimizer：优化器，决定如何更新权值参数
 7. Executor：图的前向计算与反向梯度推导
  - 类似tf.Session，当Symbol绑定executor后，当前的executor对应的图就不能再做更改了，与其他静态图框架相同
  - 用于数据并行的Executor：mxnet.executor_group.DataParallelExecutorGroup
 8. Metric：查看模型训练过程指标
 9. Callback：回调函数
  - 统计模型训练速度
 10. KVStore：跨设备的键值存储
 11. Module： 将前面所有的定义组合成一个模块使用
 
- **4、网络可视化**
- **5、保存于恢复模型**
 1. mx.callback.do_checkpoint()
 2. mx.model.load_checkpoint()
- **6、显存优化**：[memonger](https://github.com/dmlc/mxnet-memonger)
- **7、部署**：tvm、nnvm，C++或python 

# Gluon

# 分布式
 