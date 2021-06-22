import torch
import torch.nn as nn
import torch.nn.functional as  F

'''
下面为AlexNet 网络模型架构详情。
• 输入层（ Input ）：输入为3 × 224 × 224 大小图像矩阵。
• 卷积层（ Conv1 ): 96 个11× 11 大小的卷积核(每个GPU 上48 个卷积核） 。
• Pooling 层( Pool1): Max Pooling 窗口大小为2 × 2, stride=2 。
• 卷积层( Conv2): 256 个5 × 5 大小的卷积核( 每个GPU 上128 个卷积核）。
• Pooling 层( Pool2): Max Pooling 窗口大小为2 × 2, stride=2 。
• 卷积层( Conv3) : 384 个3 × 3 大小的卷积核(每个GPU 上各192 个卷积核)。
• 卷积层( Conv4): 384 个3 × 3 大小的卷积核( 每个GPU 上各192 个卷积核) 。
• 卷积层( Conv5): 256 个3 × 3 大小的卷积核(每个GPU 上各128 个卷积核) 。
• Pooling 层( Pool5): Max Pooling 窗口大小为2 × 2, stride=2 。
• 全连接层(FCl)：第一个全连接层，将第五层Pooling 层的输出连接成为一个一维向量作为该层的输入，输出4096 个神经节点。
• 全连接层(FC2)：第二个全连接层，输入输出均为4096 个神经节点。
• Softmax输出层：输出为1000 个神经节点，输出的每个神经节点单独对应图像所属分类的概率。因为在Image Net 数据集中有1000 个分类，因此设定输出维度为1000 。
alexnet原本的参数量我这个小笔记本还是带不动。。。。。

一个尺寸 a*a的特征图，经过b*b的卷积层，步幅（stride）=c，填充（padding）=d，
      请计算出输出的特征图尺寸？
答：若d等于0，也就是不填充，输出的特征图的尺寸=（a-b）/c+1
    若d不等于0，也就是填充，输出的特征图的尺寸=（a+2d-b）/c+1
'''

class Alexnet(nn.Module):
    def __init__(self,num_classes=2):
        super(Alexnet,self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256,256,kernel_size=3, padding=1)
        self.Flat = nn.Flatten()
        self.Flat1 = nn.Linear(256*6*6,4096)
        self.Flat2 = nn.Linear(4096,4096)
        self.Flat3 = nn.Linear(4096,num_classes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self,x):
        # print(x.size())
        out = self.conv1(x)
        # print(out.size())
        out=F.max_pool2d(out,2)
        # print(out.size())
        # out=self.maxpool(out)
        # print(out.size())
        out = self.conv2(out)
        out=F.max_pool2d(out,2)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out=F.max_pool2d(out,2)
        out = out.view(out.size(0), -1) # 效果和flat一样
        out = self.Flat1(out)
        out = self.Flat2(out)
        out = self.Flat3(out)

        return out




def test(net):

    x = torch.randn(2,3,224,224)
    y = net(x)
    print(y.size())
import hiddenlayer as hl
from os import path
def draw(name):
    model = eval(name)()
    ospath=os.path.split(os.path.realpath(__file__))[0].replace("\\","/")
    print(ospath)
    savepath=ospath+'/draw'
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
    im=hl.build_graph(model, torch.zeros([2,3,224,224]),transforms=transforms)
    im.save(savepath , format="jpg")
import os
from torchinfo  import summary
def summry(net):
    # params_count(net)
    summary(net,(2,3,224,224) )
if __name__ == "__main__":
    name='Alexnet'
    net = eval(name)()
    # import torchvision.models as models
    # net=models.AlexNet()
    test(net)
    summry(net)
    draw(name)

    # print(net)#可以看出网络一共有3层，两个Sequential()+avgpool
    # model_features = list(net.children())
    # print(model_features)
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