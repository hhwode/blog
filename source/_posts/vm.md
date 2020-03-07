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

# linux清理缓存
`free -m`查看内存使用情况，sysctl 命令可以临时改变某个系统参数  如：sysctl -w net.ipv4.ip_forward=1 是将forware参数临时改为1 当 service network restart后 失效.
清理步骤：
1. `sync`：因为系统在操作的过程当中，会把你的操作到的文件资料先保存到buffer中去，因为怕你在操作的过程中因为断电等原因遗失数据，所以在你操作过程中会把文件资料先缓存。所以我们在清理缓存先要先把buffe中的数据先写入到硬盘中，sync命令
2. `echo {num}`：echo 3 是清理所有缓存；echo 0 是不释放缓存；echo 1 是释放页缓存；ehco 2 是释放dentries和inodes缓存；echo 3 是释放 1 和 2 中说道的的所有缓存
```
# sync
# echo 3 > /proc/sys/vm/drop_caches
```
3. Linux清除ARP缓存
一、 `arp -n|awk '/^[1-9]/ {print "arp -d "$1}' | sh`
清除所有ARP缓存，推荐！
二、`for((ip=2;ip<255;ip++));do arp -d 192.168.0.$ip &>/dev/null;done`
清除192.168.0.0网段的所有缓存
三、`arp -d IP`
这样可以清除单一IP 的ARP缓存
注意：以上均需要root权限，尤其是最后一个，如果不再root下执行，则改为：
`arp -n|awk '/^[1-9]/ {print "arp -d "$1}' | sudo sh`
