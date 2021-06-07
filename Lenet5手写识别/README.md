# Lenet5手写识别

主要目标为：

- [ ] 寻找自己的格式
- [ ] 学习可视化

使用说明：

```python
python main.py 
# 可直接进行模型训练
```

目前功能：

- 运行 main.py  
  - 开始训练模型
  - 自动创建tensorboard所需的runs文件夹
  - 自动创建data文件夹
  - 自动创建checkpoint文件夹
- 运行model.py
  - 自动创建draw文件，内有hiddenlayer画的模型图







训练时显示数据的策略：

1、一个iter显示一次

- 太快了，高频流动，不好看数据，好处就是写起来简单直接。

- 可以换成一个batch一次比较好。

  ![image-20210605005324371](D:\Desktop\Practice\Lenet5手写识别\images\image-20210605005324371.png)

2、进度条

- 方案一的缺点都可以解决，就是写起来有点点麻烦的感觉。

![image-20210605005829373](D:\Desktop\Practice\Lenet5手写识别\images\image-20210605005829373.png)

保存模型的策略：

1、只保留最好的  if best acc <acc:  save

- 最后只有最好的模型留下了。
- resume方便





2、保留最好的 + 隔固定的epoch就保存一次

- 保留的模型多，便于回溯（缺点同理）。
- resume较第一种麻烦一点点，但解决也还是比较容易的。
