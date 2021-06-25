# Alexnet识别猫狗

## main

主要目标为：

- [x] 寻找自己的格式
- [ ] 学习可视化（CAM，gradCAM）
- [ ] 学习微调

使用说明：

```python
python main.py 
# 可直接进行模型训练
```

数据集调整为：

/data

​	/train

​		/dog01.png

目前功能：

- 继承手写识别的功能
- 自己写了个dataset类
- 优化了一下lenet里面tensorboard数据的显示
- Alexnet1是无relu的一个版本



实验：

|  |   loss   |  acc    |
| :----------------- | ---- | ---- |
| 纯conv+pool+全连接 |      |      |
| 加上relu |      |      |
| 加上Dropout |      |      |
| 加上bn层 |      |      |
| 去除之前的normalize | | |
| | | |

## 技术总结

```python
class MyDataset(torch.utils.data.Dataset):#需要继承torch.utils.data.Dataset
    def __init__(self,xxx,transform=None):
        # 初始化文件路径或文件名列表。
        # 初始化该类的一些基本参数。
        pass
    def __getitem__(self, index):
        # TODO
        #1。从文件中读取一个数据（例如，plt.imread）。
        #2。预处理数据（例如torchvision.Transform）。
        #3。返回数据对（例如图像和标签）。
        # 这里需要注意的是，第一步：read one data，是一个data
        pass
    def __len__(self):
        # 返回数据集的总大小。
```







