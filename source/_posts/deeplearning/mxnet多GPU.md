---
title: mxnet的多GPU使用
date: 2020-02-25 20:44:24
tags: introduce
category: [deeplearning,framework]
---

# 概念
mxnet自身提供指定设备的API，可以自定义数据在哪个设备上运行，但只限制在单机上，如果上升到分布式，需使用mxnet提供的PS-lite结构。

## 单机
先手动模型指定设备运行
