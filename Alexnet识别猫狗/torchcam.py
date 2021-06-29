# # simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception
# from tqdm import tqdm
# import io
# import requests
# from PIL import Image
# import torch
# from torchvision import models, transforms
# from torch.autograd import Variable
# from torch.nn import functional as F
# import numpy as np
# import cv2
# import pdb
# import os
# import gc
# finalconv_name = 'layer4' # 改成自己的层
# classes = {0:'neg', 1:'pos'} # 改成自己的对应关系
# params = list(net.parameters())[-2]	# 改成自己的层，一般可以不变
# features_blobs[-1] # 根据自己的喜好改

# def returnCAM(feature_conv, weight_softmax, class_idx, size_upsample=(384,384), use_gpu=False):
#     # generate the class activation maps upsample to size_upsample
#     bz, nc, h, w = feature_conv.shape
#     output_cam = []
#     for idx in class_idx:
#         if use_gpu:
#             cam = torch.mm(weight_softmax[idx].reshape(1,weight_softmax[idx].shape[0]), feature_conv.reshape((nc, h*w))).cpu().data.numpy()
#         else:
#             cam = torch.mm(weight_softmax[idx].reshape(1,weight_softmax[idx].shape[0]), feature_conv.reshape((nc, h*w))).data.numpy()
#         cam = cam.reshape(h, w)
#         cam = cam - np.min(cam)
#         cam_img = cam / np.max(cam)
#         cam_img = np.uint8(255 * cam_img)
#         output_cam.append(cv2.resize(cam_img, size_upsample))
#     return output_cam
    
# def camImages(net, img_dir, out_dir, size=(384,384), classes = {0:'neg', 1:'pos'}, use_gpu=False):
#     '''
#     net: 自己的模型，为了方便在训练中使用cam，所以是使用的已经加载好的
#     img_dir: 你想对哪个文件夹下的图片使用
#     out_dir: 你想保存输出图片在哪个文件夹
#     size: 跟你网络的size保持一致
#     classes: 自己类别的对应关系
#     use_gpu: 是否使用gpu来加速
#     '''
    
#     if not os.path.exists(out_dir):
#         # 判断存储文件是否存在
#         os.mkdir(out_dir)

#     # 我使用的是 ResNeSt50，最后的特征图名字为  layer4 ，可以提前print模型来查看你自己的名字
#     finalconv_name = 'layer4'
#     if use_gpu:
#         net = net.cuda()
#     else: 
#     	net = net.cpu()
#     net.eval()

#     features_blobs = []
#     def hook_feature(module, input, output):
#         features_blobs.append(output.detach().data)
#     # hook函数，使用handle作为返回是为了删除它，不然会内存泄漏
#     handle = net._modules.get(finalconv_name).register_forward_hook(hook_feature)

#     # 一般都是这个分布
#     normalize = transforms.Normalize(
#     mean=[0.485, 0.456, 0.406],
#     std=[0.229, 0.224, 0.225]
#     )

#     preprocess = transforms.Compose([
#     transforms.Resize(size),
#     transforms.ToTensor(),
#     normalize
#     ])

#     # 开始单张图片的检测
#     for _img in tqdm(os.listdir(img_dir)):
#         img_path = os.path.join(img_dir, _img)  #获取图片路径
#         img_pil = Image.open(img_path).convert('RGB')   #读取图片

#         img_tensor = preprocess(img_pil)    #转为tensor
#         img_variable = Variable(img_tensor.unsqueeze(0))
#         if use_gpu:
#             logit = net(img_variable.cuda()).detach().cpu()
#         else:
#             logit = net(img_variable).detach()
        
#         params = list(net.parameters())[-2] # 这个 -2 要注意，改成自己网络的对应层数，全连接层前面那一层，一般的网络是 -2
#         weight_softmax = np.squeeze(params.detach().data)        

#         h_x = F.softmax(logit, dim=1).data.squeeze()
#         probs, idx = h_x.sort(0, True)
#         probs = probs.numpy()
#         idx = idx.numpy()
#         # features_blobs[-1]表示的是你之前的 finalconv_name 那一层的特征的最后的输出
#         CAMs = returnCAM(features_blobs[-1], weight_softmax, [idx[0]], size_upsample=size, use_gpu=use_gpu)

#         img = cv2.imread(img_path)
#         height, width, _ = img.shape
#         heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
#         result = heatmap * 0.3 + img * 0.6
#         cv2.imwrite(os.path.join(out_dir, _img), result)  
#     # 这个时候删除 hook
#     handle.remove()


# if __name__ == "__main__":
#     net_path = './checkpoints/resnest50_v7_momentum/40.pt'
#     net = torch.load(net_path)
#     use_gpu = False
#     img_dir = './data_zf/neg'
#     out_dir = './temp_img/neg'
#     size = (368, 368)
#     # 自己的类别对应关系
#     classes = {0:'neg', 1:'pos'}

#     camImages(net, img_dir, out_dir, size=size, classes=classes, use_gpu=use_gpu)
# Define your model

import os
import torch
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchcam.cams import SmoothGradCAMpp

from torchvision.models import resnet18
from model import Alexnet
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# 定义model

# model = resnet18(pretrained=True).eval()

ospath=os.path.split(os.path.realpath(__file__))[0].replace("\\","/")
model = Alexnet().eval()
modelname=type(model).__name__
checkpoint = torch.load(ospath+'/checkpoint/' + modelname +'.pth' )
model.load_state_dict(checkpoint['net'])
print(model)
# 设置CAM提取器
cam_extractor = SmoothGradCAMpp(model)# 没有指定名称的话，默认返回卷积层的最后一个
cam_extractor = SmoothGradCAMpp(model,'conv5')# 可以指定对应卷积层的名称，名字可以通过print获得

# 设置输入
img = read_image(r"D:\Desktop\Practice\Alexnet识别猫狗\data\train\cat.0.jpg")
# 为选择的模型进行预处理
input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
input_tensor = input_tensor.unsqueeze(0)
print(input_tensor.shape)
# 处理数据并喂入模型
out = model(input_tensor)
# 通过传递类索引和模型输出来检索CAM
activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

import matplotlib.pyplot as plt
# 现在原始的CAM
plt.imshow(activation_map.numpy()); plt.axis('off'); plt.tight_layout(); plt.show()

from torchcam.utils import overlay_mask

# Resize the CAM and overlay it 调整CAM的大小并将其覆盖
result = overlay_mask(to_pil_image(img), to_pil_image(activation_map, mode='F'), alpha=0.5)
# Display it
plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()