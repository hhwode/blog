---
title: ArcFace_Additive Angular Margin Loss for Deep Face Recognition
date: 2020-03-04
tags: introduce
category: [deeplearning,face]
---
2019年

# Three Pass

## first

**列出**：

 	1. 标题、摘要、介绍
     
     - 标题：ArcFace： Additive Angular Margin Loss for Deep Face Recognition，使用附加角度间隔的损失来进行深度人脸识别，从损失来进行改进是最近几年比较流行的一种人脸识别的方法。
     
     - 摘要：深度学习我们可以考虑到最后一层全连接层是一种特征提取的方式，之后有些进行分类的都是基于loss来设定的，比如softmax就是看对应位置哪个输出概率最大，所以前面NN层都是用于特征提取，没有多大修改空间，最终起决定作用的还是loss，**只要能定义一种合适的loss function，就可以提高区分效果。**
     
       比较常见的loss function如下：
     
       > 1、Centre loss：该损失在欧式空间上弥补了NN提取的特征与对应类别的距离，实现了类内更加紧密。在一定程度上压缩了类内聚集。
       >
       > 2、SphereFace：
     
       最近比较热门的研究方向是：
     
       作者也从这个方向进行研究，提出ArcFace（Additive Angular Margin Loss）来获取更高区分度的特征，是基于弧面与超球面进行测试距离的方法，可以说几何解释比较清晰。
     
       作者对比了超过10种人脸识别的基准实验，ArcFace在计算和性能上都具有优势，更难能可贵的是开源所有实验内容，以供重现。
     
     - 介绍：使用DCNN来提取脸部特征表示是人脸识别的一种方法。有些类似词向量，是不是。训练方式有两种：（**类内距离越小，类间距离越大**）
     
       > 1. 通过所有类别数据加在一起训练，使用softmax分类器，就是一种普通分类训练方法；1）缺点-全连接层的权重大小依据类别数线性增长。2）只能学到已有类别的数据，无法处理新类别。
       >
       >    改善方式：1）Centre loss
       >
       >    如softmax工作原理：
       >
       >    ```
       >    只有三个类别，训练第三个类别数据：
       >    iter1 model->概率 [0.2, 0.2, 0.5]->完成训练
       >    ```
       >
       >    
       >
       > 2. 直接学习一个embedding特征，用于后续人脸聚类来进行人脸识别，但需要结合特定的loss，如triplet-loss-based；1）组合多，大数据量下需迭代多次。2）semi-hard sample不利于模型训练。
       >
       >    改善方式：1）Sphereface，2）CosFace，3）ArcFace
       >
       >    如CosFace工作原理：
       >
       >    ```
       >    只有三个类别，训练第三个类别数据：
       >    iter1 model->概率 [0.2,0.2,0.5]->强化训练，类别概率减0.5->[0.2,0.2,0.0]->未完成继续训练
       >    iter2 model->概率 [0.2,0.2,0.8]->强化训练，类别概率减0.5->[0.2,0.2,0.3]->完成训练 
       >    ```
       >
       >    
     
         这块介绍ArcFace的四个有点：Engaging、Effective、Easy、Efficient
     
  2. 章节

  3. 数学原理

  4. 结论

  5. 文献

**回答**：

	1. 论文类型
 	2. 相关内容
 	3. 正确性
 	4. 贡献点
 	5. 相关文献
 	6. 是否继续



## second



## third



