---
title: Linux安装horovod
date: 2020-02-13 20:44:24
tags: install
category: deeplearning
---

# 在线安装
- **1、升级gcc**
	tar -jxvf gcc-4.9.4.tar.bz2
	cd gcc-4.9.4
	./contrib/download_prerequisites 
	[如果连接失败，无法下载的话，就打开此文件，手动下载下面5个文件，然后将文件放在gcc根目录，再屏蔽contrib/download_prerequisites文件里面的wget操作，再重新执行一次./contrib/download_prerequisites。这样的话，后面编译gcc时，这几个依赖库会自动先编译，不用自动手动一个个编译。
	cloog-0.18.1.tar.gz
	gmp-4.3.2.tar.bz2
	isl-0.12.2.tar.bz2
	mpc-0.8.1.tar.gz
	mpfr-2.4.2.tar.bz2]
	
	mkdir build-gcc
	../configure --enable-checking=release --enable-languages=c,c++ --disable-multilib
	make -j 8 (-j利用多核处理器加快速度，机器核数*2)
	make install
	
	在~/.bash_profile配置库文件和头文件路径:
	```
	export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64/:$LD_LIBRARY_PATH
	export C_INCLUDE_PATH=/usr/local/include/:$C_INCLUDE_PATH
	export CPLUS_INCLUDE_PATH=/usr/local/include/:$CPLUS_INCLUDE_PATH
	```
	gcc --version 或 g++ -version

- **2、安装Python3.6**
	cd Python-3.6.6
	mkdir /usr/local/python3.6
	./configure --prefix=/usr/local/python3.6
	make
	make install
	make uninstall
	
	ln -s /usr/local/python3.6/bin/python3.6  /usr/bin/python3
	ln -s /usr/local/python3.6/bin/pip3  /usr/bin/pip3

- **3、安装TensorFlow、pytorch、mxnet**
	pip3 install tensorflow-1.14.0-cp36-cp36m-manylinux1_x86_64.whl -i http://mirrors.aliyun.com/pypi/simple  --trusted-host mirrors.aliyun.com
	pip3 install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

- **4、安装OpenMPI**
    tar -jxvf 
	cd openmpi-4.0.0
	./configure --prefix=/usr/local/openmpi
	make all install -j32
	
	在~/.bash_profile配置库文件和头文件路径:
	```
	export PATH="/home/$USERNAME/openmpi/bin:$PATH"
	export LD_LIBRARY_PATH="/home/$USERNAME/openmpi/lib/:$LD_LIBRARY_PATH"
	```

- **5、安装horovod**
```
    pip3 install --ignore-installed horovod -i http://mirrors.aliyun.com/pypi/simple  --trusted-host mirrors.aliyun.com
	or
	pip3 install --ignore-installed horovod -i https://pypi.tuna.tsinghua.edu.cn/simple/
```
	
	HOROVOD_MPICXX_SHOW=$(/usr/local/openmpi/bin/mpicxx -show) pip3 install dist/horovod-0.19.0.tar.gz
	
	HOROVOD_MPICXX_SHOW=$(/usr/local/openmpi/bin/mpicxx -show) pip3 install --ignore-installed horovod -i http://mirrors.aliyun.com/pypi/simple  --trusted-host mirrors.aliyun.com