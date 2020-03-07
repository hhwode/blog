---
title: PyTorch RPC
date: 2020-03-03
tags: introduce
category: [deeplearning,distributedTraining]
---

# PyTorch RPC
## 1 介绍
PyTorch在2020年1月16日发布的1.4.0版本中开始支持分布式模型并行（验证性），该版本提供了一个分布式RPC框架来支持分布式模型并行训练，它允许远程运行函数和引用远程对象，而不需要拷贝实际数据，并提供autograd和优化器API来跨RPC进行反向传播和参数更新。
在1.4.0版本后提供torch.distributed.rpc包来支持分布式模型并行，现阶段还是验证性的，不稳定，API也会修改（看过1.5.0a0+ad3f4a3版，API稍有不同）。当然，用户可以依据源码进行适配。
Pytorch目前支持分布式数据并行和单机模型并行，作者对强化学习学习策略和模型量考虑，所以才提供RPC框架来支持分布式模型并行。我们主要关注第二点原因，即：
当模型太大，不能在单个机器上进行模型并行时，就需要将模型划分到多个机器上。
此外，也可以使用这个RPC框架来自定义PS结构。以下内容基于PyTorch 1.4.0版本介绍进行整理。
## 2 框架
深度学习的分布式模型并行难在如下两点，需要不同进程协助工作：
- 1、模型划分后，前后有依赖关系，需要传递数据；
- 2、跨机器的反向传播与梯度更新策略如何制定。
### 2.1torch.distributed.rpc
PyTorch提供的 torch.distributed.rpc包有以下几部分组成：RPC、RRef、distributed autograd、distributed optimizer；前两个是接收和发送数据，后两个是类似在本地执行反向传播与梯度更新操作。
### 2.1.1RPC
Remote Procedure Call（RPC）支持使用给定的参数在指定的目标节点上运行函数，并获取返回值或创建返回值的引用。可以认为指定函数到远端节点计算，而本地只需要获取远端节点返回的输出值即可，能使我们像操作本地计算一样使用远程节点计算。
现阶段主要函数如下：
- 1、init_rpc()：初始化RPC框架，RRef框架和distributed autograd，后端默认使用gloo通信，也支持mpi、nccl。
- 2、rpc_sync()：同步操作，有阻塞，如果用户代码在没有返回值的情况下无法继续，则使用同步API。
- 3、rpc_async()：异步操作，异步就需要一个额外的future来存储远端返回值，以供调用。
- 4、remote()：异步操作并返回对远程返回值的引用，比如远端创建一些东西，但不需要将其获取给本地，也就是本地不需要知道远端某些操作，也不会调用。
- 5、get_worker_info()：获取指定worker信息
- 6、shutdown()：关闭RPC代理
API：
torch.distributed.rpc.init_rpc(name, backend=BackendType.PROCESS_GROUP, rank=-1, world_size=None, rpc_backend_options=None)

torch.distributed.rpc.rpc_sync(to, func, args=None, kwargs=None)
torch.distributed.rpc.rpc_async(to, func, args=None, kwargs=None)
torch.distributed.rpc.remote(to, func, args=None, kwargs=None)
### 2.1.2RRef
Remote Reference（RRef）用作指向本地或远程对象的分布式共享指针。用该种方式创建的变量可以本地和远端使用，但每个RRef只有一个拥有者，而对象只存在于该拥有者上，而持有RRef的非拥有者可以通过显式请求从拥有者那里获得对象的副本。	比如该对象在A创建，但B也持有，那么B访问该对象时，就会从A那里拷贝一份，所以此处会产生数据传输。Distributed Optimizer就是依据此种功能实现。
主要函数：
- 1、is_owner()：判断是否引用拥有者
- 2、local_value()：如果是拥有者，就返回对应本地值，否则发生异常
- 3、owner()：返回拥有者信息
- 4、to_here()：会阻塞，将引用拥有者的值复制到本地使用，如果当前节点时拥有者就直接返回该值。
### 2.1.3Distributed Autograd
计算梯度模块，用于分布式发送和接收梯度。分布式Autograd将所有节点上涉及前向传递的本地Autograd引擎缝合在一起，并在后向传递期间自动与它们接触以计算梯度。PyTorch将分布式前向反向操作串在一起，不需要用户自己实现，按理更新时就是每个节点更新自己那部分。
主要函数：
- 1、torch.distributed.autograd.context：前向和反向传播时需要的标识，只会创建一个
- 2、torch.distributed.autograd.backward：
- 3、torch.distributed.autograd.get_gradients
该部分是重点，单机在前向传播时，PyTorch会创建autograd图，用于反向传播；而分布式需要跟踪所有RPC操作，为了创建autograd图，其中涉及到机器间的通信，所以distributed autograd比autograd多了一些函数协助，如send和recv函数。
### 2.1.4Distributed Optimizer
分布式优化器需要指定待优化的参数RRef列表，在每个不同的RRef拥有者上创建一个Optimizer对象，并在调用该对象的step函数时，更新对应参数。跟2.1.3Distributed Autograd类似，都是每个节点有一份，而在主节点进行封装成一份使用，所以才让用户不感知。
主要函数：
- 1、torch.distributed.optim.DistributedOptimizer：将远程引用分散到每个节点，存在1）无法保证同一时间客户端都执行完整个前向后向优化，2）无法保证节点的执行顺序？
- 2、step()：拥有参数的每个节点都会调用torch.optim.Optimizer.step()来更新参数，并且会阻塞，直到所有worker更新完。
## 3示例

```python
# 在给定的RRef上调用方法的辅助函数
def _call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)

# 该函数用于在rref的拥有者上运行方法并使用RPC取回结果，远程调用使用
def _remote_method(method, rref, *args, **kwargs):
    return rpc.rpc_sync(
        rref.owner(),
        _call_method,
        args=[method, rref] + list(args),
        kwargs=kwargs)

# 为给定的本地模块中的每个参数创建一个RRef，并返回一个RRef列表，但不清楚是# 否远端所有参数都会生成一个单独的RRef在本地，如果是，那本地就拥有所有的参  # 数；如果不是，那么多一些RRef引用，还是能接受
def _parameter_rrefs(module):
    param_rrefs = []
    for param in module.parameters():
        param_rrefs.append(RRef(param))
    return param_rrefs
```
## 4总结
对该框架的一些理解性总结，是否合适，又或者能否支持


