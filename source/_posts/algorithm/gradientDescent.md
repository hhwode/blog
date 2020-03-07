---
title: Gradient Descent
date: 2020-03-01
tags:
---
模仿都是参照原品，完成一次模仿后与原品对比，查看两者之间的误差程度，重复操作，直到模仿品与原品误差最小，才会将模仿品拿出来把玩，这个使自己创造出的与原品进行误差最小化的思想与梯度下降法思想类似。

## 1 目标

机器学习说到底，就是假设模型类型已找出，并找一个目标函数，一直优化到最小值，然后所求参数就是我们所需，将参数带回模型中就是求取到的模型。

以线性回归为例，假设只从一个维度来观察样本数据，即只有一维特征，可以假设模型类型是线性类型、指数类型等，如
$$ Y=wx+b $$
或者
$$ Y=wx^2+b $$
诸如此类的模型，这里选用最简单的`Y=wx+b`。

我们能获取到的就是观察值，比如`x=1`时`Y`等于多少这类观察值，既有特征，又有结果，只是不知道是什么模型导致出现这类结果的。比如想知道重力加速度是多少，我们能知道释放物体的高度（特征），可以知道物体到达地面的时间（观察真实值），这样一组数据要多少可以实验多少，用观察值来确定加速度（参数），就是机器学习适用范围。

前面说了这么多，汇总就是一句话，使找到的模型输出值与观察真实值误差越小越好（误差就是`观察真实值-模型输出值`），一模一样就是找到了真正的模型。将结果汇总为公式如下：
$$ Target = \sum_{r=1}^n(y^{'}-y). $$
因`y'`与`y`有大小之分，两者的差值会有正负，但观察到的每个样本都是相互独立的，此处的正负值会将样本间联系起来，此种情况不是我们需要的，所以一般会写出绝对值形式或者平方形式，以去除掉每个样本计算误差的正负号。此时误差可以变成如下形式
$$ Target = \sum_{r=1}^n|y^{'}-y| \;\quad  or \;\quad Target = \sum_{r=1}^n(y^{'}-y)^2. $$
绝对值在某些情况下比较适用，但其有正负之分，**不利于计算**。一般适用平方项，**因其计算容易，并且还存在导数**。到此，我们找到了需要优化的目标。
$$ Target = \sum_{r=1}^n(y^{'}-y)^2. $$
该目标也叫成本函数（Cost Function），是整个训练集上误差的平均值；还有一个是损失函数（Loss Function），是针对单个训练示例的误差计算。

## 2 求解

前面找到了需要优化得目标，但如何求解到能是目标最优化的参数呢？这就是梯度下降法（gradient descent），通过找**哪条路**和**走多远**来到底最小值处。

梯度下降法不是闭式解，只是通过一步步逼近最优点的近似解。如下公式，每一步都是以 $\alpha$ 倍的梯度来对参数进行惩罚。
$$ \theta^1 = \theta^0-\alpha\Delta J(\theta). $$

## 3 例子

以线性回归为例，只有一个特征，为了方便可视化，针对pytorch编程，可以认为分6步走：

- 1. **获取到观察到的数据**

- 2. **建立网络**

- 3. **确定成本函数cost function**

- 4. **确定优化器**

- 5. **训练**

- 6. **评估**

  如下是pytorch代码展示。

```python
    import torch
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    from torch import nn
    from torch import optim
    import matplotlib.pyplot as plt
    import numpy as np
    # 1、获取到观察到的数据
    x = torch.randn((100, 1), dtype=torch.float32, requires_grad=True)
    y = 3.13 * x + 0.01159
    x = x + torch.normal(0, 0.1, (100, 1), dtype=torch.float32)

    train_loader = DataLoader(dataset=TensorDataset(x, y),
                              batch_size=5,
                              shuffle=False)
    # 2、建立网络
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.net = nn.Linear(1, 1, bias=True)

        def forward(self, input_features):
            return self.net(input_features)
    # torch的网络更像定义一个函数，指定输入，给我输出结果即可，不关心里面的其他操作
    model = Net().to("cuda")

    # 3、确定成本函数cost function
    def mean_error(in_tensor, out_tensor):
        return torch.pow((in_tensor - out_tensor), 2).sum() / in_tensor.size()[0]
    # loss = nn.MSELoss()
    loss = mean_error
    # print(x.size()[0])
    # 4、确定优化器
    optimizer = optim.SGD(model.parameters(), lr=0.015)
    # 5、训练
    for epoch in range(100):
        for batch_idx, (features, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(features.to("cuda"))
            loss_output = loss(output, labels.to("cuda"))
            # 从loss往前进行传播
            loss_output.backward(retain_graph=True)
            optimizer.step()
        print("epoch", epoch, "loss", loss_output.cpu().data.numpy())
    for param in model.cpu().parameters():
        print(type(param.data), param.size(), param.detach().numpy())
    # 6、评估
    result = []
    for features, labels in train_loader:
        output = model(features)
        result.extend(output.cpu().detach().numpy())
        # print(output.data.numpy())
    result = np.asarray(result)
    # print('result', result)
    # print(result.shape)
    raw_x = x.detach().numpy()
    # print(raw_x)
    # print(raw_x.shape)
    # draw
    # print(y.detach().numpy().shape)
    plt.scatter(raw_x[:, 0], y.detach().numpy(), c='r')
    plt.scatter(raw_x[:, 0], result, c='b')
    plt.plot(raw_x[:, 0], result)
    plt.show()
```

给特征随机加入均值为0，方差为0.1的噪声，真实模型的参数为`w=3.13`, `b=0.01159`，训练100个epoch时的最优参数为`w=3.1186929`, `b=0.00647835`，与真实模型参数误差较小。

如下图，红色为真实`x,y`对应的点，蓝色与直线为预测出来的`x,y`对应点，基本符合，结合简单即最优的策略，找出来的模型是可以实际使用。

![](/images/algorithm/gd/1.png "result")