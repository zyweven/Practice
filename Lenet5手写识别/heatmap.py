from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' 
def draw_CAM(model,img_path,save_path,resize=227,isSave=False,isShow=False):
    # 图像加载&预处理
    img=Image.open(img_path).convert('RGB')
    loader = transforms.Compose([transforms.Resize(size=(resize,resize)),transforms.ToTensor()]) 
    img = loader(img).unsqueeze(0) # unsqueeze(0)在第0维增加一个维度
    
    # 获取模型输出的feature/score
    model.eval() # 测试模式，不启用BatchNormalization和Dropout
    feature=model.features(img)
    output=model.classifier(feature.view(1,-1))
    
    # 预测得分最高的那一类对应的输出score
    pred = torch.argmax(output).item()
    pred_class = output[:, pred]

    # 记录梯度值
    def hook_grad(grad):
        global feature_grad
        feature_grad=grad
    feature.register_hook(hook_grad)
    # 计算梯度
    pred_class.backward()
    
    grads=feature_grad # 获取梯度
    
    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1)) # adaptive_avg_pool2d自适应平均池化函数,输出大小都为（1，1）

    # 此处batch size默认为1，所以去掉了第0维（batch size维）
    pooled_grads = pooled_grads[0] # shape为[batch,通道,size,size],此处batch为1，所以直接取[0]即取第一个batch的元素，就取到了每个batch内的所有元素
    features = feature[0] # 取【0】原因同上
    
    ########################## 导数（权重）乘以相应元素
    # 512是最后一层feature的通道数
    for i in range(len(features)):
        features[i, ...] *= pooled_grads[i, ...] # features[i, ...]与features[i]效果好像是一样的，都是第i个元素
    ##########################
    
    # 绘制热力图
    heatmap = features.detach().numpy()
    heatmap = np.mean(heatmap, axis=0) # axis=0,对各列求均值，返回1*n
 
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    # 可视化原始热力图
    if isShow:
        plt.matshow(heatmap)
        plt.show()
        
    img = cv2.imread(img_path)  # 用cv2加载原始图像
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子
    # 将图像保存到硬盘
    if isSave:
        cv2.imwrite(save_path, superimposed_img)  
        print('su')
    # 展示图像
    if isShow:
        superimposed_img/=255
        plt.imshow(superimposed_img)



if __name__ ==  '__main__':
    import torchvision.models as models
    model=models.alexnet()
    draw_CAM(model,'C:/Users/jiede/Pictures/123.jpg','热力图1.png',isSave=True,isShow=True)
