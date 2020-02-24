---
title: mxnet
date: 2020-02-18 20:44:24
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

# 常用API

## gpu()

## cpu()

## symbol()
符号式编程，通过mx.symbol来进行调用网络模块，可以定义：
1. mx.symbol.Variable:类似tf.placeholder功能，作为一个占位符
2. mx.symbol.Convolution
3. mx.symbol.Activation
4. mx.symbol.BatchNorm

## Context()
通过输入设备不同来建立设备，mx.Context('gpu')，其实等价于：
1. mx.Context('gpu') == mx.gpu()
2. mx.Context('cpu') == mx.cpu()

## ctx()

## Symbols()

## AttrScope()

```
def test_ctx_group():
    with mx.AttrScope(ctx_group='stage1'):
        data = mx.symbol.Variable('data')
        fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
        act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")

    set_stage1 = set(act1.list_arguments())
    print(list(act1.list_arguments()))
    with mx.AttrScope(ctx_group='stage2'):
        fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
        act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
        fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=10)
        fc3 = mx.symbol.BatchNorm(fc3)
        mlp  = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')

    set_stage2 = set(mlp.list_arguments()) - set_stage1
    print(list(mlp.list_arguments()))
    print('set_stage2', set_stage2)
    group2ctx = {
        'stage1' : mx.cpu(0),
        'stage2' : mx.gpu(0)
    }
    texec = mlp.simple_bind(mx.cpu(0),
                            group2ctx=group2ctx,
                            data=(1,200))

    for arr, name in zip(texec.arg_arrays, mlp.list_arguments()):
        if name in set_stage1:
            assert arr.context == group2ctx['stage1']
            print('stage1', arr)
        else:
            assert arr.context == group2ctx['stage2']
            print('stage2', arr)
```

## 建立一个网络
以线性回归网络为例，其就是一个单层网络
### mxnet定义
1. 定义输入，明确特征与标签，形状，数量
```
num_inputs = 2
num_examples = 1000
true_w=[2, -3.4]
true_b=4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0]*features[:,0]+true_w[1]*features[:,1]+true_b
# noise
labels += nd.random.normal(scale=1, shape=labels.shape)
```
2. 构建迭代器
```
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        j = nd.array(indices[i:min(i+batch_size,num_examples)])
        yield features.take(j), labels.take(j)
```

3. 建网络
```
w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
b=nd.zeros(shape=(1,))
# create gradient
w.attach_grad()
b.attach_grad()
# 网络定义
def linreg(x, w, b):
    return nd.dot(x,w)+b
# 损失函数定义
def square_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape))**2/2
# 优化器定义
def sgd(params, lr, batch_size):
    for param in params:
        param[:]=param -lr*param.grad / batch_size
```
4. 构建执行流程
```
lr = 0.03
num_epochs=3
net=linreg
loss = square_loss
for epoch in range(num_epochs):
    for x,y in data_iter(batch_size, features, labels):
        with autograd.record():
		    # 计算当前batch的损失值，用于反向传播
            l = loss(net(x,w,b), y)
		# 损失值或误差反向传播，方法嵌入在mxnet中，直接调用，相当于nd.sum(loss).backward()
        l.backward()
		# 优化参数
        sgd([w,b],lr,batch_size)
    train_l = loss(net(features,w,b),labels)
    print('epoch %d, loss %f' %(epoch+1, train_l.mean().asnumpy()))
```
5. 问题
	- 建立网络时，定义loss值为什么要y.reshape(y_hat.shape)：有可能网络预测出来的值与输入y形状不同，依赖网络定义。
	- 如果样本个数不能被batch整除，有啥影响：最后一个batch size值就会不够

### gluon定义
gluon实现由自己的输入类型
1. 定义输入，明确特征与标签，形状，数量
```
num_inputs = 2
num_examples = 1000
true_w=[2, -3.4]
true_b=4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0]*features[:,0]+true_w[1]*features[:,1]+true_b
# noise
labels += nd.random.normal(scale=1, shape=labels.shape)
```
2. 转换数据，创建迭代器
```
# gluon
from mxnet.gluon import data as gdata
batch_size = 10
dataset = gdata.ArrayDataset(features, labels)
train_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
```
3. 定义网络
```
# 1. 定义网络
from mxnet.gluon import nn
net = nn.Sequential()
# 输出1个值
net.add(nn.Dense(1))
# 2. 初始化网络参数
from mxnet import init
net.initialize(init.Normal(sigma=0.01))
# 3. 定义损失函数
from mxnet.gluon import loss as gloss
loss = gloss.L2Loss()
# 定义优化器，使用什么优化算法
from mxnet import gluon
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.03})
```
4. 构建执行流程
```
num_epochs = 3
for epoch in range(num_epochs):
    for x,y in train_iter:
	    # 
        with autograd.record():
		    # 计算batch下网络损失值
            l = loss(net(x), y)
		# 反向传播，相当于nd.sum(loss).backward()
        l.backward()
		# 优化参数值，将前面batch size的loss进行归一化，trainer.step(batch_size)相当于除以batch size
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print('epoch %d, loss %.4f' %(epoch+1, l.mean().asnumpy()))
```
可通过`net[0].weight.data()`，`net[0].bias.data()`来查看第0层的权值与偏差，`net[0].weight.grad()`来查看权值梯度值
5. 问题
	- 如果将 l = loss(net(X), y) 替换成 l = loss(net(X), y).mean()，我们需要将 trainer.step(batch_size) 相应地改成 trainer.step(1)。这是为什么呢：因为mean已经对loss进行了平均，而trainer.step(batch_size)作用在优化参数时会除以batch_size来做平均
# Gluon

# 分布式

# Question

1. 问题描述
```
Traceback (most recent call last):
  File "test_mxnet_model_parallel_02.py", line 37, in <module>
    test_ctx_group(kv)
  File "test_mxnet_model_parallel_02.py", line 27, in test_ctx_group
    data=(1,200))
  File "/usr/local/python3.6/lib/python3.6/site-packages/mxnet/symbol/symbol.py", line 1541, in simple_bind
    ctx_map_dev_types.append(val.device_typeid)
AttributeError: 'list' object has no attribute 'device_typeid' 
```
解答：bind doesn't support multi-gpu. Use mx.mod.Module instead，代码用了simple_bind