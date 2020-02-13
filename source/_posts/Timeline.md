---
title: Timeline
date: 2020-02-07 20:44:24
tags: introduce
category: deeplearning
---
深度学习模型越来越大，数据越来越多，当训练慢时如何定位原因，考虑：通信，GPU占用，GPU利用率等原因
其中timeline就是一个用于记录模型训练时每个operator耗时情况，主要关注哪部分在CPU上运行，哪部分在GPU运行，是否可以将CPU的放到GPU上，通过减少**关键路径**的耗时, 提升性能
能结合一个例子说明更好
# Timeline介绍
本文只是对timeline的简单介绍与使用。
现在深度学习模型越来越大，数据越来越多，训练性能分析尤为重要，如何定位原因呢，可考虑几个几个方面：通信，GPU占用，GPU利用率；这些方面有对应的工具进行统计，如nvidia-smi， nmon。而TensorFlow有一个工具Timeline，horovod也集成了。
Timeline是一个用于记录模型训练时每个operator耗时情况，显示模型在CPU和GPU运行情况，通过Timeline分析减少关键路径的耗时, 可以提升性能。
## 组成
Timeline网上介绍也少，只解释说是一种时间记录手段，TensorFlow主要还是通过Profile API来进行分析，其中Timeline就是Profile的一部分。一般只取一个epoch情况，不需要合并多个epoch情况。
加载情况，打开Chrome，在地址栏键入chrome://tracing/，通过load按钮导入 timeline.json 文件，就可以看到每个ops的运行时间和设备信息了。如下：
![](/images/timeline/concept.png "concept")
MEMCPYHtoD operation操作，H代表Host，理解为cpu或者本机内存，D代表Device，这里表示GPU
GPU:0/stream  代表的是 cuda执行的时间，可以看到各个 operation是 有先后顺序的
以Grad结束的操作其实是梯度操作，最后梯度的流出方向是  apply 也就是把梯度应用到 参数变量上。

View Options中的Flow events会将op的依赖关系用箭头显连接

1. 最顶端的左边Record、Save、Load是可操作按钮，右边View Options可显示ops的依赖过程，如图中中部的有箭头部分
2. 左边一栏的/device /gpu是CPU和GPU设备信息，具体分析如下：
1)/device:GPU:0/stream:all Compute (所有GPU花费总时间,GPU使用情况)
2)/gpu:0/context#0/stream#1:Kernel Compute(GPU计算情况)
3)/gpu:0/context#0/stream#2:MemcpyHtoD Compute(CPU拷贝数据到GPU计算情况)
4)/gpu:0/context#0/stream#3:MemcpyDtoH Compute(GPU拷贝数据到CPU计算情况)
5)/host:CPU Compute(CPU计算情况)
6)/job:localhost/replica:0/task:0/device:CPU:0 Compute(cpu使用情况)
7)/job:localhost/replica:0/task:0/device:GPU:0 Compute(cpu把任务发到gpu上)
2) , 3), 4)加起来就是1)的情况，但5)跟6)有什么区别，7)跟其他又有什么区别
3. 右边是对应ops在该设备下的运行时间
4. 点击一个ops，就会出现最下面部分，该ops的具体信息：名字、类型、输入输出、开始时间与持续时间等信息
5. 最右边的File Size Stats，Metrics待分析···

## 一些概念
概念：
1. /device:GPU:0/stream:all Compute (所有GPU花费总时间,GPU使用情况)
2. /gpu:0/context#0/stream#1:Kernel Compute(GPU计算情况)
3. /gpu:0/context#0/stream#2:MemcpyHtoD Compute(CPU拷贝数据到GPU计算情况)
4. /gpu:0/context#0/stream#3:MemcpyDtoH Compute(GPU拷贝数据到CPU计算情况)
5. /host:CPU Compute(CPU计算情况)
6. /job:localhost/replica:0/task:0/device:CPU:0 Compute(cpu使用情况)
7. /job:localhost/replica:0/task:0/device:GPU:0 Compute(cpu把任务发到gpu上)
2,3,4加起来就是1的情况，但5跟6有什么区别，7跟其他又有什么区别

# 使用
MNSIT数据集的DNN模型，模型比较简单，就三个隐藏层，一个softmax输出层。Timeline只是记录了某一轮(即某个epoch)的训练结果。
1.定义run_options 和 run_metadata, 用于保存op的属性
```
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata() 
```
2.保存
```
fetched_timeline = timeline.Timeline(run_metadata.step_stats)
chrome_trace = fetched_timeline.generate_chrome_trace_format()
with open('timeline.json' % i, 'w') as f:
f.write(chrome_trace)
```
## TensorFlow纯CPU情况
从Timeline可以看出，模型训练时主要在CPU上，没有GPU的信息，说明未使用到GPU，总的耗时在18ms左右。
![](/images/timeline/1.png "timeline")

## 指定网络在GPU上
也是会用到GPU

## CPU+GPU情况
从Timeline可以看出，模型训练时计算主要在GPU上，总的耗时在5ms左右，比纯CPU快。
但：**相同代码，使用到GPU为什么结果跟只使用CPU不相同，而且连op都不一样**
![](/images/timeline/2.png "timeline")

# keras使用timeline
因此不同于传统的Tensorflow 方式，需要在 model.compile()中增加profile相关配置。

> 代码部分：
> 1. 定义 run_options 和 run_metadata, 用于保存op的属性
```
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata() 
……
```
> 2. 将  run_options 和 run_metadata 添加至 model的 compile方法
```
model.compile(
	……
	options=run_options,
	run_metadata=run_metadata
)
```
> 3. 在 timeline.json中保存 run_metadate
```
from tensorflow.python.client import timeline
tl = timeline.Timeline(run_metadata.step_stats)
ctf = tl.generate_chrome_trace_format()
with open(os.path.join(logdir, 'timeline.json'), 'a+') as f:
	f.write(ctf)
```
一般只取一个epoch情况，不需要合并多个epoch情况
# horovod使用timeline
Horovod也封装了Timeline，只需指定Timeline输出文件位置即可，比TensorFlow、Keras简单，不用嵌入代码。
使用mpirun命令跑timeline，[官网链接](https://github.com/horovod/horovod/blob/master/docs/mpirun.rst)
```
$ HOROVOD_TIMELINE=/path/to/timeline.json mpirun -np 4 -x HOROVOD_TIMELINE python train.py
```

使用horovodrun命令跑timeline
```
$ horovodrun -np 4 --timeline-filename /path/to/timeline.json python train.py
```
只是一个工具的初步认识，深入分析需要实验相结合。
未知Horovod跑pytorch、mxnet是否也可以用该Timeline工具