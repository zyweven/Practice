import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
import random

random.seed(1)
'''
猫狗数据集虽然下载下来自然就有train，test数据集之分。
但其中test数据集是没有标签的。所以训练时只有把train中的数据进行划分了。
那样读取数据又有了两种方式。一种是提前划好，又单独的弄train和test文件夹（硬划分）
一种是就在dataset里面通过random来进行划分（软划分）
这里为了省事我选择软划分
'''
class CatDogDataset(Dataset):
    
    def __init__(self,file_path,flag='',splitrate=0.8, rng_seed=2,transform=None):
        # TODO
        # 1. Initialize file path or list of file names.
        self.rng_seed=rng_seed
        random.seed(self.rng_seed)
        self.splitrate=splitrate
        self.path=os.path.join(file_path,'train')
        self.imgname=os.listdir(self.path)
        self.transform=transform
        random.shuffle(self.imgname)
        idx= int(len(self.imgname) * self.splitrate)  # 25000* 0.8 = 20000

        if flag=='train':
            self.data=self.imgname[:idx]
        elif flag=='test':
            self.data=self.imgname[idx:]

        self.imgpath=[os.path.join(self.path,img) for img in self.data ]
        self.label=[img.split(".")[0] for img in self.data]
        self.img_labels = [0 if n.startswith('cat') else 1 for n in self.label] # 把猫变成 0 狗变成1 
        #忘记这步的话,label在变为batch的时候不能转为tensor只能变为tuple:('dog','cat'....,'dog')，tuple不能to到cuda上



    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        #这里需要注意的是，第一步：read one data，是一个data
        img = Image.open(self.imgpath[index]).convert('RGB')
        label=self.img_labels[index]
        # print(img)
        if self.transform is not None:
            try:
                img = self.transform(img)   # 在这里做transform，转为tensor等等
            except:
                img.show()
            else:
                return img,label
        # print(img)
        return img,label

    def __len__(self):
         # You should change 0 to the total size of your dataset.
        return len(self.data)



# class CatDogTest(Dataset):
    
#     def __init__(self,file_path,transform=None):
#         self.path=os.path.join(file_path,'test')
#         self.data=os.listdir(self.path)
#         self.imgpath=[os.path.join(self.os.path,img) for img in self.data ]
#         self.label=[img.split(".")[0] for img in self.data]


#     def __len__(self):
#         pass

#     def __getitem__(self, index):
#         pass
# if __name__ == "__main__":
#     file_path=r'D:\Desktop\Practice\Alexnet识别猫狗\data'
#     # print(os.listdir(file_path))
#     path=os.path.join(file_path,'train')
#     print(path)
#     # print(os.listdir(path))
#     print(len(os.listdir(path)))
#     data=os.listdir(path)
#     imgpath=[os.path.join(path,img) for img in data ]
#     # print(imgpath)
#     # plt.imshow(imgpath[0]) plt好像不支持中文路径
#     a=random.randint(2,25000)

#     path=os.path.join(file_path,'train')
#     splitrate=0.8
#     imgname=os.listdir(path)
#     random.shuffle(imgname)
#     idx= int(len(imgname) * splitrate)  # 25000* 0.8 = 20000
#     data=imgname[:idx]
#     imgpath=[os.path.join(path,img) for img in data ]
#     label=[img.split(".")[0] for img in data]
#     img=Image.open(imgpath[a])
#     img.show()
#     # label=[img.split(".")[0] for img in data]
#     print(label[a])
#     img = Image.open(imgpath[1]).convert('RGB')
#     print(type(img))