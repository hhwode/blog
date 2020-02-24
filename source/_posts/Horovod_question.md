---
title: Horovod 框架使用问题集
date: 2020-02-19 22:44:24
tags: [deepLearning,distributedTraining]
category: introduce
---

# 执行
## horovodrun执行
常用参数
1. /usr/local/python3.6/bin/horovodrun --verbose -np 2 -H master:1,clone02:1 python3 hvd_mnist_mod_02.py

## mpirun执行

# 执行问题
1. 问题描述
Traceback (most recent call last):
  File "/usr/local/python3.6/lib/python3.6/runpy.py", line 193, in _run_module_as_main
    "__main__", mod_spec)
  File "/usr/local/python3.6/lib/python3.6/runpy.py", line 85, in _run_code
    exec(code, run_globals)
  File "/usr/local/python3.6/lib/python3.6/site-packages/horovod/run/task_fn.py", line 67, in <module>
    _task_fn(index, driver_addresses, settings)
  File "/usr/local/python3.6/lib/python3.6/site-packages/horovod/run/task_fn.py", line 27, in _task_fn
    driver_addresses, settings.key, settings.verbose)
  File "/usr/local/python3.6/lib/python3.6/site-packages/horovod/run/driver/driver_service.py", line 35, in __init__
    match_intf=match_intf)
  File "/usr/local/python3.6/lib/python3.6/site-packages/horovod/run/common/service/driver_service.py", line 153, in __init__
    match_intf=match_intf)
  File "/usr/local/python3.6/lib/python3.6/site-packages/horovod/run/common/util/network.py", line 174, in __init__
    'Linux.'.format(service_name=service_name, addresses=addresses))
horovod.run.common.util.network.NoValidAddressesFound: Horovodrun was unable to connect to horovodrun driver service on any of the following addresses: {'lo': [('127.0.0.1', 30616)], 'ens33': [('192.168.110.150', 30616)], 'virbr0': [('192.168.122.1', 30616)]}.

One possible cause of this problem is that horovodrun currently requires every host to have at least one routable network interface with the same name across all of the hosts. You can run "ifconfig -a" on every host and check for the common routable interface. To fix the problem, you can rename interfaces on Linux.

解决办法：
	- 1. 可能ssh无密码登录未设置
	- 2. 可能防火墙未关闭

