---
title: MXNet分布式
date: 2020-02-15 20:44:24
tags: introduce
category: [deeplearning,distributedTraining]
---

# 1 介绍
MXNet是DMLC社区开源的深度学习框架，具有如下特点：
	1. 接口清晰，易用
	2. 显存利用与多卡并行效率高
	3. 部署成熟，PC、移动
	4. 支持多语言python、R、matlab、JavaScript等
	5. 具有静态图编程接口MXNet和动态图编程接口gluon（MXNet和gluon可以看出类似TensorFlow和Keras的关系，但gluon是动态图，与其对应的是pytorch）
# 2 分布式
数据很大，分布式已经是趋势，节点之间共享参数是必须，面临三个挑战：
	1. 如何利用高带宽来获取共享参数
	2. 很多机器学习是顺序的，同步和高延迟的阻碍影响了性能
	3. 高容错率
MXNet[[1]](https://mxnet.incubator.apache.org/api/faq/multi_device)[2]基于参数服务器结构上进行了改进，提出的ps-lite分布式训练结构。
## 2.1 ps-lite
ps-lite[[3]](https://www.cnblogs.com/heguanyou/p/7868596.html) [[4]](https://ps-lite.readthedocs.io/en/latest/how_to.html)包含三种角色：Worker、Server、Scheduler（默认基于服务器级别作为Worker）。具体关系如下图：
 ![](/images/mxnet-distribute/1.png "ps-lite")
	- Worker节点负责计算参数，并发参数push到Server，同时从Server pull参数回来。
	- Server节点负责管理Worker节点发送来的参数，并“合并”，之后供各个Worker使用。
	- Scheduler节点负责管理Worker节点和Server节点的状态，Worker与Server之间的连接是通过Scheduler的。
ps-lite支持两种分布式计算模式，数据并行和模型并行，并同时支持同步和异步参数更新。
	- 数据并行，指的是每个设备存储完整模型副本的情况，每个设备使用数据集的不同部分进行工作，设备共同更新共享模型。
	- 模型并行，不同的设备被分配了学习模型不同部分的任务。 目前，MXNet仅支持单机中的模型并行。
## 2.2 相关概念

### 2.2.1 进程类型
MXNet中有三种进程类型，这些进程之间相互通信，完成模型的训练。
	- Worker：Worker节点实际上在一批训练样本上进行训练。 在处理每个批次之前，Workers从服务器上拉出权重。 Worker还会在每个batch处理后向服务器发送梯度(gradient)。 根据训练模型的工作量，在同一台机器上运行多个工作进程可能不是一个好主意，MXNet默认以机器作为Worker的粒度，减少了参数push/pull操作。
	- Server：可以有多个Servers存储模型的参数，并与Workers进行通信。 Server可能与Worker进程同处一处,也可能不。如果只有一个Server，模型的所有参数都存储在该Server上，如果有多个，会依据key值进行随机划分模型参数，每个Server存储一部分。
	- Scheduler(调度器)：只有一个Scheduler。Scheduler的作用是配置集群。这包括等待每个节点启动以及节点正在监听哪个端口之类的消息。 然后Scheduler让所有进程知道集群中的其他节点的信息，以便它们可以相互通信。优先启动的进程。
### 2.2.2 Kvstore
分布式训练关键部分，Servers将参数存储为K-V形式，以便Workers向Servers进行push/pull操作。
需要指定给训练器，
	1. MXNet情况如下，只有Module才有分布式参数kvstore：
	```
	mod = mx.mod.Module(mlp)  # mlp是最后分类层
	mod.bind(data_shapes=train_iter.provide_data,
			 label_shapes=train_iter.provide_label)
	mod.init_params()
	kv = mx.kvstore.create('dist_sync')
	mod.fit(train_iter, eval_data=val_iter,optimizer_params={'learning_rate':0.01, 'momentum': 0.9},num_epoch=2, kvstore=kv)
	```
	或者：
	```
	mod.fit(train_iter, eval_data=val_iter,optimizer_params={'learning_rate':0.01, 'momentum': 0.9},num_epoch=2, kvstore="dist_sync")
	```

	2. gluon情况下：
	```
	trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate' : lr},kvstore='dist_sync')
	```
	   
	或者如下，可以用kv.rank知道哪个worker

	```
	kv = mx.kvstore.create('dist_sync')
	trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate' : lr}, kvstore=kv)
	```

### 2.2.3 Keys的分配
每个Server不一定存储所有的key或全部的参数数组。 参数分布在不同的Server上。 哪个Server存储特定的keys是随机决定的。 KVStore透明地处理不同服务器上的keys分配。 它确保当一个keys被拉取时，该请求被发送到的服务器具有对应value。 如果某个keys的值非常大，则可能会在不同的服务器上分片。这意味着不同的服务器拥有不同部分的value。
我们之前跑的是单个Server情况，可以尝试多个Server，看其参数划分后，通信开销是否减少，性能是否有提高。
### 2.2.4 切分训练数据集
单个Worker的数据并行训练，我们可以使用mxnet.gluon.utils.split_and_load来切分数据迭代器(data iterator)提供的一批样本，然后将该批处理的每个部分加载到将处理它的设备上。
分布式训练的情况下，我们需要在开始时将数据集分成n个部分，以便每个Worker获得不同的部分。然后，每个Worker可以使用split_and_load再次将数据集的这部分划分到单个机器上的不同设备上。
### 2.2.5 分布式训练的不同模式
使用不同类型的kvstore可以启用不同的分布式训练模式：
	- dist_sync：同步分布式训练，当每个节点上使用多个GPU时使用，此模式在CPU上聚合梯度并更新权重。
	- dist_async：异步分布式训练。
	- dist_sync_device：与dist_sync相同，当每个节点上使用多个GPU时使用，此模式在GPU上聚合梯度并更新权重。
	- dist_async_device：与dist_sync_device相似，但是是异步模式。
### 2.2.6 梯度压缩
当通信费用昂贵，并且计算时间与通信时间的比例较低时，通信可能成为瓶颈。 在这种情况下，可以使用梯度压缩来降低通信成本，从而加速训练[[5]](https://mxnet.incubator.apache.org/api/faq/gradient_compression)。
怎么压缩的
如何使用
## 2.3 如何使用
MXNet提供了一个执行分布式训练的脚本，该工具是在mxnet安装位置下的tools/launch.py，可以方便执行，并介绍了手动启动分布式训练的方法，设置对应环境变量即可[[1]](https://mxnet.incubator.apache.org/api/faq/multi_device)。
### 2.3.1 launch.py工具
MXNet提供了一个脚本工具/ launch.py，以便于开展分布式训练工作。这支持各种类型的集群资源管理器，如ssh，mpirun，yarn和sge。launch.py工具只是针对服务器级别为Worker，因为其只起了一个进程，如果要起多个进程，如何指定一个进程对应一个GPU，比较难设置。
比如在单机上运行脚本：
```
#python image_classification.py --dataset cifar10 --model vgg11 --num-epochs 1
```
此示例的分布式训练，可执行以下操作： 如果包含脚本image_classification.py的mxnet目录可供集群中的所有计算机访问（例如，如果它们位于网络文件系统上），则可以运行：
```
#../../tools/launch.py -n 3 -H hosts --launcher ssh python3 mnist.py --kvstore dist_sync
```
如果包含脚本的目录不能从集群中的其他机器访问，那么我们可以将当前目录同步到所有机器。
```
#../../tools/launch.py -n 3 -H hosts --launcher ssh --sync-dst-dir /tmp/mxnet_job/ python3 mnist.py --kvstore dist_sync
```
launch.py提交分布式训练工具参数：
	**-n** 表示要启动的worker节点的数量。
	**-s** 表示要启动的server节点的数量。未指定时，默认等于Worker节点的数量
	**--launcher** 表示通信模式。选项有：
		ssh 如果机器可以通过ssh进行通信而无需密码。 这是默认启动模式。
		mpi 使用Open MPI时开启
		sge 适用于Sun Grid引擎
		yarn 适用于Apache yarn
		local 用于在同一本地计算机上启动所有进程。 这可以用于调试。
	**-H** 需要主机文件的路径,该文件包含集群中机器的IP，一行一个IP。这些机器应能够在不使用密码的情况下相互通信。 此文件仅适用于启动程序模式为ssh或mpi时
	**--sync-dst-dir** 将所有主机上的一个目录的路径指向当前将被同步的工作目录。此选项仅支持ssh启动模式。
### 2.3.2 手动启动
MXNet使用环境变量将不同的角色分配给不同的进程，并让不同的进程查找调度程序，launch.py工具就是将这个手动分配的工作进行封装，不需要用户自己手动指定。
包含如下几个环境变量需要设置：
	**DMLC_ROLE** ：指定进程的角色。 这可以是server、worker或scheduler。注意，只有一个scheduler。 当DMLC_ROLE设置为server或scheduler后，这些进程在导入mxnet时启动。
	**DMLC_PS_ROOT_URI** ：指定scheduler的IP
	**DMLC_PS_ROOT_PORT** ：指定scheduler侦听的端口
	**DMLC_NUM_SERVER** ：指定群集中有多少个server节点
	**DMLC_NUM_WORKER** ：指定群集中有多少个worker节点

启动方式，比如启动1个scheduler，2个server，2个worker：
```
#DMLC_ROLE=scheduler DMLC_PS_ROOT_URI=127.0.0.1 DMLC_PS_ROOT_PORT=9092 DMLC_NUM_SERVER=2 DMLC_NUM_WORKER=2 python3 mnist.py --kv-store dist_sync

#DMLC_ROLE=server DMLC_PS_ROOT_URI=127.0.0.1 DMLC_PS_ROOT_PORT=9092 DMLC_NUM_SERVER=2 DMLC_NUM_WORKER=2 python3 mnist.py --kv-store dist_sync

#DMLC_ROLE=server DMLC_PS_ROOT_URI=127.0.0.1 DMLC_PS_ROOT_PORT=9092 DMLC_NUM_SERVER=2 DMLC_NUM_WORKER=2 python3 mnist.py --kv-store dist_sync

#DMLC_ROLE=worker DMLC_PS_ROOT_URI=127.0.0.1 DMLC_PS_ROOT_PORT=9092 DMLC_NUM_SERVER=2 DMLC_NUM_WORKER=2 python3 mnist.py --kv-store dist_sync

#DMLC_ROLE=worker DMLC_PS_ROOT_URI=127.0.0.1 DMLC_PS_ROOT_PORT=9092 DMLC_NUM_SERVER=2 DMLC_NUM_WORKER=2 python3 mnist.py --kv-store dist_sync
```
## 2.4 实用工具
### 2.4.1 其他环境变量使用
1. **PS_VERBOSE**：输出通信消息
	PS_VERBOSE=1: logging connection information
	PS_VERBOSE=2: logging all data communication information
2. **DMLC_INTERFACE**：指定通信网卡
	DMLC_INTERFACE=br0
更多环境变量设置见[[6]](https://mxnet.incubator.apache.org/api/faq/env_var)。
### 2.4.2 Profile
类似timeline的工具，可生成profile.json[[7]](https://mxnet.apache.org/api/python/docs/tutorials/performance/backend/profiler.html)文件，在Chrome浏览器上输入chrome://tracing/进行加载该json文件。
1. **MXNET_PROFILER_AUTOSTART=1**：指定环境变量运行mxnet程序，可以不需要修改代码，将把整个程序执行情况统计，文件会变成很大。如下图跑一个epoch的mnist分类情况。
 ![](/images/mxnet-distribute/2.png "profile")
2. **代码指定**
代码指定的方式，可自定义统计部分，文件容量相对较少。
创建profile配置
```
from mxnet import profiler
profiler.set_config(profile_all=True,
               aggregate_stats=True,
               continuous_dump=True,
               filename='profile_output.json')
```
指定统计范围
```
profiler.set_state('run') # 开始统计
run_training_iteration(*next(itr)) # 训练
mx.nd.waitall() 
profiler.set_state('stop') # 结束统计
profiler.dump()
```

# 3 优化
MKL


**Reference**
[1] https://mxnet.incubator.apache.org/api/faq/multi_device
[2] Scaling Distributed Machine Learning with the Parameter Server
[3] https://www.cnblogs.com/heguanyou/p/7868596.html
[4] https://ps-lite.readthedocs.io/en/latest/how_to.html
[5] https://mxnet.incubator.apache.org/api/faq/gradient_compression
[6] https://mxnet.incubator.apache.org/api/faq/env_var
[7] https://mxnet.apache.org/api/python/docs/tutorials/performance/backend/profiler.html

**附录**：
一、MXNet建立模型
MXnet中定义好symbol、写好dataiter并且准备好data之后，就可以开开心的去训练了。一般训练一个网络有两种常用的策略，基于model的和基于module的。
1. **使用Model构建模型**
从官方文档里面拿出来的代码看一下（Model形式不能用于分布式训练，没找到kvstore参数接口）：
```
# 1、configure a two layer neural network
    data = mx.symbol.Variable('data')
    fc1 = mx.symbol.FullyConnected(data, name='fc1', num_hidden=128)
    act1 = mx.symbol.Activation(fc1, name='relu1', act_type='relu')
    fc2 = mx.symbol.FullyConnected(act1, name='fc2', num_hidden=64)
    softmax = mx.symbol.SoftmaxOutput(fc2, name='sm')
# 2、create a model using sklearn-style two-step way
#创建一个model
   model = mx.model.FeedForward(
         softmax,
         num_epoch=num_epoch,
         learning_rate=0.01)
#开始训练
    model.fit(X=data_set)
```
具体的API参照http://mxnet.io/api/python/model.html。Model形式可定制化不强，不常用，一般用Module构建模型。
2. **使用Module构建模型**
Module有四种状态：
	1. 初始化状态，就是显存还没有被分配，基本上啥都没做的状态。
	2. bind，在把data和label的shape传到Bind函数里并且执行之后，显存就分配好了，可以准备好计算能力。
	3. 参数初始化。就是初始化参数
	4. Optimizer installed 。就是传入SGD，Adam这种optimuzer中去进行训练　
构建代码（Module也是可以用于分布式训练的）：
```
    import mxnet as mx
    # construct a simple MLP
    data = mx.symbol.Variable('data')
    fc1  = mx.symbol.FullyConnected(data, name='fc1', num_hidden=128)
    act1 = mx.symbol.Activation(fc1, name='relu1', act_type="relu")
    fc2  = mx.symbol.FullyConnected(act1, name = 'fc2', num_hidden = 64)
    act2 = mx.symbol.Activation(fc2, name='relu2', act_type="relu")
    fc3  = mx.symbol.FullyConnected(act2, name='fc3', num_hidden=10)
    out  = mx.symbol.SoftmaxOutput(fc3, name = 'softmax')
 
    # construct the module
    mod = mx.mod.Module(out)
    
    # mod.bind的操作是在显卡上分配所需的显存，所以需要把data_shapehe label_shape传递给他
    mod.bind(data_shapes=train_dataiter.provide_data,
            label_shapes=train_dataiter.provide_label)
    
    mod.init_params()
    mod.fit(train_dataiter, eval_data=eval_dataiter,
          optimizer_params={'learning_rate':0.01, 'momentum': 0.9},
          num_epoch=n_epoch)
```
参考：https://www.cnblogs.com/daihengchen/p/6506386.html



