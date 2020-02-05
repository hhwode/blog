---
title: Using Hexo to deploy blog
date: 2020-02-03 20:44:24
tags: web
category: introduce
keywords: hexo, github, nodejs
---
一直想在GitHub中搭建自己的博客，可是始终未清楚需要什么，现在看很多博客的模板，直接借鉴过来使用。

## 步骤

### 1、安装Nodejs和配置好Nodejs环境

### 2、安装Git和配置好Git环境

### 3、安装Hexo
1、安装命令如下
``` bash
$ npm install hexo -g
```
2、查看是否安装成功
``` bash
$ hexo -v
```
3、初始化hexo
到自定义安装目录下，执行
``` bash
$ hexo init
```
看到后面的"Start blogging with Hexo！"就是安装成功
4、安装所需组件
``` bash
$ npm install
$ npm install hexo-deployer-git --save
```
5、测试
``` bash
$ hexo g
```
6、本地服务启动
``` bash
$ hexo s
```
在浏览器打开http://localhost:4000/ 查看

### 4、关联Hexo与Github，设置Git的user name和email
1、ssh配置
``` bash
$ ssh-keygen -t rsa -C "your@mail.com"
```
输入eval "$(ssh-agent -s)"，添加密钥到ssh-agent
输入ssh-add ~/.ssh/id_rsa，添加生成的SSH key到ssh-agent
登录Github，点击头像下的settings，添加ssh
新建一个new ssh key，将id_rsa.pub文件里的内容复制上去
输入ssh -T git@github.com，测试添加ssh是否成功。如果看到Hi后面是你的用户名，就说明成功了
2、配置Deployment，在其文件夹中，找到_config.yml文件，修改repo值
![deploy配置](/images/hexo2github/deploy.png "Title")
repo值是你在github项目里的ssh（右下角）
### 5、新建blog

``` bash
$ hexo new post "博客名"
```
在文件夹_posts目录下将会看到已经创建的文件

### 6、远端部署

``` bash
$ hexo g -d
```

### 7、Something

![初始化hexo后文件列表](/images/hexo2github/source.png "初始化hexo后文件列表")
![文档列表，一个文件一个文档](/images/hexo2github/blog.png "文档列表，一个文件一个文档")
![每个文档使用Markdown语法编写](/images/hexo2github/context.png "每个文档使用Markdown语法编写")

More info: [Deployment](https://hexo.io/docs/one-command-deployment.html)
