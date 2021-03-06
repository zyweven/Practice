import torch
import torch.nn as nn
import torch.nn.functional as  F

class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5,self).__init__()
        self.convC1 = nn.Conv2d(3,6,kernel_size=5)# 3表示输入图片是3通道的，6表示输出6个特征图，也就是6个卷积核
        self.convC3 = nn.Conv2d(6,16,kernel_size=5)
        self.Flat = nn.Flatten()
        self.FlatC51 = nn.Linear(400,120)
        self.FlatC52 = nn.Linear(120,84)
        self.FlatC53 = nn.Linear(84,10)
        
    
    def forward(self,x):
        out=self.convC1(x)
        out=F.max_pool2d(out,2)
        out=self.convC3(out)
        out=F.max_pool2d(out,2)
        # out = out.view(out.size(0), -1) # 效果貌似和Flatten一样
        out = self.Flat(out)
        # print(out.shape)#(2,400) 其中2是bathsize,400是前面三个通道相乘得到的16*5*5
        out=self.FlatC51(out)
        out=self.FlatC52(out)
        out=self.FlatC53(out)
        return out

# class LeNet(nn.Module):
#     def __init__(self):
#         super(LeNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1   = nn.Linear(16*5*5, 120)
#         self.fc2   = nn.Linear(120, 84)
#         self.fc3   = nn.Linear(84, 10)

#     def forward(self, x):
#         out = F.relu(self.conv1(x))
#         out = F.max_pool2d(out, 2)
#         out = F.relu(self.conv2(out))
#         out = F.max_pool2d(out, 2)
#         out = out.view(out.size(0), -1)
#         out = F.relu(self.fc1(out))
#         out = F.relu(self.fc2(out))
#         out = self.fc3(out)
#         return out


def test(net):
    # net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())
import hiddenlayer as hl
from os import path
def draw(name):
    # name='VGG11'
    model = eval(name)()
    # model = VGG('VGG11')
    ospath=os.path.split(os.path.realpath(__file__))[0].replace("\\","/")
    print(ospath)
    savepath=ospath+'/draw'
    # savepath="D:/Desktop/pytorch-cifar-master/pytorch-cifar-master/draw"
    savepath=path.join(savepath,name)
    print(savepath)
    transforms = [
        hl.transforms.Fold("Conv > BatchNorm > Relu", "MaxPool"),
        # Fold Conv, BN, RELU layers into one
        hl.transforms.Fold("LeakyRelu > MaxPool > Conv > BatchNorm", "ConvBnRelu"),
        # Fold Conv, BN layers together
        hl.transforms.Fold("Conv > BatchNorm > LeakyRelu", "ConvBn"),
        # Fold bottleneck blocks
        hl.transforms.Fold("""
            ((ConvBnRelu > ConvBnRelu > ConvBn) | ConvBn) > Add > Relu
            """, "BottleneckBlock", "Bottleneck Block"),
        # Fold residual blocks
        hl.transforms.Fold("""ConvBnRelu > ConvBnRelu > ConvBn > Add > Relu""",
                        "ResBlock", "Residual Block"),
        # Fold repeated blocks
        hl.transforms.FoldDuplicates(),
    ]
    im=hl.build_graph(model, torch.zeros([2,3,32,32]),transforms=transforms)
    im.save(savepath , format="jpg")
import os
from torchinfo  import summary
def summry(net):
    # params_count(net)
    summary(net,(2,3,32,32) )
if __name__ == "__main__":
    name='Lenet5'
    net = eval(name)()
    test(net)
    # draw(name)
    summry(net)
    print(net)
    model_features = list(net.children())
    print(model_features)
    # print(model_features[0][3])#取第0层Sequential()中的第四层
    # print()
    # for index,layer in enumerate(model_features[0]):
    #     print(layer)

# ==========================================================================================
# Layer (type:depth-idx)                   Output Shape              Param #
# ==========================================================================================
# Lenet5                                   --                        --
# ├─Conv2d: 1-1                            [2, 6, 28, 28]            456
# ├─Conv2d: 1-2                            [2, 16, 10, 10]           2,416
# ├─Flatten: 1-3                           [2, 400]                  --
# ├─Linear: 1-4                            [2, 120]                  48,120
# ├─Linear: 1-5                            [2, 84]                   10,164
# ├─Linear: 1-6                            [2, 10]                   850
# ==========================================================================================