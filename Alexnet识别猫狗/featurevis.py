'''

zyw
使用双线性插值的方法实现特征可视化
最终输出模型第k层的所有特征图可视化结果(比如该层有六个特征图那就有六个图)，图像在feature_map_save/k文件夹里面,
特征图显示是黑白的
'''

import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import imageio
plt.rcParams['font.sans-serif']=['STSong']
# 导入模型
# import torchvision.models as models
from model import Lenet5
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# #1.模型查看
# print(model)#可以看出网络一共有3层，两个Sequential()+avgpool
# model_features = list(model.children())
# print(model_features[0][3])#取第0层Sequential()中的第四层
# for index,layer in enumerate(model_features[0]):
#     print(layer)



#2. 导入数据
# 以RGB格式打开图像
# Pytorch DataLoader就是使用PIL所读取的图像格式
# 建议就用这种方法读取图像，当读入灰度图像时convert('')
def get_image_info(image_dir):
    image_info = Image.open(image_dir).convert('RGB')#是一幅图片
    # 数据预处理方法
    image_transform = transforms.Compose([
        transforms.Resize((32,32)),
        # transforms.CenterCrop(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_info = image_transform(image_info)#torch.Size([3, 224, 224])
    image_info = image_info.unsqueeze(0)#torch.Size([1, 3, 224, 224])因为model的输入要求是4维，所以变成4维
    return image_info#变成tensor数据


#2. 获取第k层的特征图
'''
args:
k:定义提取第几层的feature map
x:图片的tensor
model_layer：是一个Sequential()特征层
'''
def get_k_layer_feature_map(model_layer, k, x):
    with torch.no_grad():
        for index, layer in enumerate(model_layer):#model的第一个Sequential()是有多层，所以遍历
            x = layer(x)#torch.Size([1, 64, 55, 55])生成了64个通道
            print('x=')
            print(x.shape)
            if k == index:
                return x


#  可视化特征图
def show_feature_map(feature_map):#feature_map=torch.Size([1, 64, 55, 55]),feature_map[0].shape=torch.Size([64, 55, 55])
                                                                         # feature_map[2].shape     out of bounds
    feature_map = feature_map.squeeze(0)#压缩成torch.Size([64, 55, 55])
    
    #以下4行，通过双线性插值的方式改变保存图像的大小
    feature_map =feature_map.view(1,feature_map.shape[0],feature_map.shape[1],feature_map.shape[2])#(1,64,55,55)
    upsample = torch.nn.UpsamplingBilinear2d(size=(256,256))#可通过这里调整输出图像的大小
    feature_map = upsample(feature_map)
    feature_map = feature_map.view(feature_map.shape[1],feature_map.shape[2],feature_map.shape[3])
    
    feature_map_num = feature_map.shape[0]#返回通道数
    row_num = np.ceil(np.sqrt(feature_map_num))#8
    plt.figure()
    for index in range(1, feature_map_num + 1):#通过遍历的方式，将64个通道的tensor拿出
        plt.subplot(row_num, row_num, index)
        # 灰色
        plt.imshow(feature_map[index - 1], cmap='gray')#feature_map[0].shape=torch.Size([55, 55])
        #将上行代码替换成，可显示彩色 
        # 彩色
        # plt.imshow(transforms.ToPILImage()(feature_map[index - 1]))#feature_map[0].shape=torch.Size([55, 55])
        plt.axis('off')
        if not os.path.exists(ospath+'/feature_map_save'+'//'+str(k)):
            os.mkdir(ospath+'/feature_map_save'+'//'+str(k))
        imageio.imsave( ospath+'/feature_map_save//'+str(k)+'//'+str(index) + ".png", feature_map[index - 1])
    plt.show()



if __name__ ==  '__main__':

# 模型载入
    # model = models.alexnet(pretrained=True)
    model = Lenet5()
    ospath=os.path.split(os.path.realpath(__file__))[0].replace("\\","/")
    modelname=type(model).__name__
    checkpoint = torch.load(ospath+'/checkpoint/' + modelname +'.pth' )
    model.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    startepoch = checkpoint['epoch']
#1.模型查看
    print(model)#可以看出网络一共有3层，两个Sequential()+avgpool
    # model_features = list(model.children())
    # print(model_features)
    # print(model_features[3])#取第0层Sequential()中的第四层
    # print()
    # for index,layer in enumerate(model_features[0]):
    #     print(layer)

    image_dir = r"D:\Desktop\123.jpg"
    k = 2    # 定义提取第几层的feature map
    image_info = get_image_info(image_dir)
    if not os.path.exists(ospath+'/feature_map_save'):
        os.mkdir(ospath+'/feature_map_save')

    # model = models.alexnet(pretrained=True)
    model_layer= list(model.children())
    #! model_layer=model_layer[0]
    model_layer=model_layer# 这里选择model的第一个Sequential()
    print(model_layer[k])
    print()
    feature_map = get_k_layer_feature_map(model_layer, k, image_info)
    show_feature_map(feature_map)


# # 如果model用的是Sequential()这样的写法
# 如：
# Sequential(
#   (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
#   (1): ReLU(inplace=True)
#   (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
#   (4): ReLU(inplace=True)
#   (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (7): ReLU(inplace=True)
#   (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (9): ReLU(inplace=True)
#   (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (11): ReLU(inplace=True)
#   (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
# )
#  则 需要使用model_layer=model_layer[0]来选中 Sequential

# 没有Sequential
# Lenet5(
#   (convC1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
#   (convC3): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
#   (Flat): Flatten(start_dim=1, end_dim=-1)
#   (FlatC51): Linear(in_features=400, out_features=120, bias=True)
#   (FlatC52): Linear(in_features=120, out_features=84, bias=True)
#   (FlatC53): Linear(in_features=84, out_features=10, bias=True)
# )这里面就只有卷积层，也就只能可视化卷积