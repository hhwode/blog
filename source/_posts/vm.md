---
title: vm虚拟机
date: 2020-02-19 20:44:24
tags: install
category: deeplearning
---

# 修改hostname
1. 适用于网络中，此时需要修改配置文件   vim  /etc/sysconfig/network
```
HOSTNAME=master
```
2. 修改host文件

3. 用命令：`hostnamectl set-hostname {newhost-name}`

4. 编辑配置文件：vim  /etc/hostname  将里面的值替换

以上方式都必须重启才能生效

# 克隆Linux修改网络
重启网卡提示Bringing up interface eth0:  Device eth0 does not seem to be present,delaying initialization.                    [FAILED]:
1. 因为克隆的机器没有正确的mac,UUID信息冲突导致的，首先将/etc/udev/rules.d/70-persistent-net.rules文件清空或删除
2. 然后将网卡配置文件/etc/sysconfig/network-scripts/ifcfg-eth0的uuid和hwaddr这两行删除
网卡名ifcfg-eth0可修改，但文件名与文件内容需一致
3. vim /etc/sysconfig/network-scripts/ifcfg-eth0  （这里的网卡名称根据自己的网卡名称来写）#增加一条  `ARPCHECK=no`
4. 重启网卡就能正常重启网络了：
`/etc/init.d/network restart`
`service network restart`
`systemctl start network.service`
	- 1.service命令
	service命令其实是去/etc/init.d目录下，去执行相关程序
```
	# service命令启动redis脚本
	service redis start
	# 直接启动redis脚本
	/etc/init.d/redis start
	# 开机自启动
	update-rc.d redis defaults
	其中脚本需要我们自己编写
```
	- 2.systemctl命令
	systemd是Linux系统最新的初始化系统(init),作用是提高系统的启动速度，尽可能启动较少的进程，尽可能更多进程并发启动。systemd对应的进程管理命令是systemctl

# 防火墙
1. 查看所有服务：`systemctl list-unit-files|grep enabled`
2. 禁用防火墙：`systemctl stop firewalld.service`, `systemctl disable firewalld.service`
3. 查看防火墙状态：`systemctl status firewalld.service`

启动一个服务：systemctl start firewalld.service
关闭一个服务：systemctl stop firewalld.service
重启一个服务：systemctl restart firewalld.service
显示一个服务的状态：systemctl status firewalld.service
在开机时启用一个服务：systemctl enable firewalld.service
在开机时禁用一个服务：systemctl disable firewalld.service
查看服务是否开机启动：systemctl is-enabled firewalld.service;echo $?
查看已启动的服务列表：systemctl list-unit-files|grep enabled

