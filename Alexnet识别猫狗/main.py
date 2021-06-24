'''
2021年6月17日 20:02:13
zyw
Alexnet 识别猫狗

'''
from numpy.core.fromnumeric import resize
import torch
from torch import optim
from torch._C import device
from torch.autograd.grad_mode import no_grad
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

import os
from datetime import datetime
import time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from torch.utils.tensorboard import SummaryWriter
from model import Alexnet
from utils import progress_bar
from dataset import CatDogDataset
train_iter=0
test_iter=0
def train(epoch):
    net.train()
    train_loss=0
    total = 0
    correct=0
    global train_iter
    for batch_idx,(input,label) in enumerate(train_loadata):
        # print(label)
        if device =='cuda':
            input=input.to(device)
            label=label.to(device)
        # plt.imshow(input)
        optimizer.zero_grad()
        output=net(input)
        iter_loss = loss(output,label)
        iter_loss.backward()
        optimizer.step()
        batchsize=label.size(0)
        # tensorboard 
        train_iter += batch_idx
        train_loss+=iter_loss.item()
        _,predicate = output.max(1) #返回每一行中最大的元素并返回索引，返回了两个数组.predicate返回的就是索引 max(0)是返回每列的。
        total+=label.size(0) 
        # print(label.size(0)) # label.size(0)表示batchsize
        correct+=predicate.eq(label).sum().item()
        acc=100*correct/total
        writer.add_scalar('train/loss',iter_loss,batch_idx)
        progress_bar(batch_idx,len(train_loadata),'Loss:{:0.4f} |  LR:{:0.6f}'.format(iter_loss.item(),optimizer.param_groups[0]['lr']))
        # print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
    train_loss=train_loss/len(train_data)
    writer.add_scalar('trainepoch/losss',train_loss,epoch)
    print('Training Epoch: {epoch} Loss: {:0.4f}\tLR: {:0.6f}'.format(
        iter_loss.item(),
        optimizer.param_groups[0]['lr'],
        epoch=epoch,
        # trained_samples=n_iter,
        # total_samples=int(len(train_loadata.dataset)/batchsize)+(len(train_loadata.dataset)%batchsize ==1)
    ))


def test(epoch):
    print('Evaluating Network.....')
    global test_iter
    global best_acc
    net.eval()
    modelname=type(net).__name__
    test_loss =0
    correct =0
    start=time.time()
    with torch.no_grad():
        for batch_idx ,(input,label) in enumerate(test_loadata):
            # print(label)
            if device == 'cuda':
                input=input.to(device)
                label = label.to(device)
            output = net(input)
            iter_loss = loss(output,label).item()
            _,predicted = output.max(1)
            batchsize = predicted.size(0)
            test_loss +=iter_loss
            test_iter += batch_idx
            correct+=predicted.eq(label).sum().item()
            iter_acc=correct/ ((batch_idx+1)*batchsize)*100
            writer.add_scalar('test/loss',iter_loss,test_iter) 
            writer.add_scalar('test/acc',correct/((batch_idx+1)*batchsize),test_iter)
            progress_bar(batch_idx,len(test_loadata),'Loss:{:0.4f} |  Acc:{:0.6f}'.format(iter_loss,iter_acc))
        finish = time.time()
        acc=correct / len(test_loadata.dataset)*100
        test_loss=test_loss/len(test_loadata)
        writer.add_scalar('testepoch/losss',test_loss,epoch) 
        writer.add_scalar('testepoch/accc',acc,epoch)
        print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
            epoch,
            test_loss / len(test_loadata.dataset),
            acc,
            finish - start
        ))
        
    if acc>best_acc:
        print('Saving .....')
        state = {
            'net':net.state_dict(),
            'acc':acc,
            'epoch':epoch,
        }
        if not os.path.isdir(ospath+'/checkpoint'):
            os.mkdir(ospath+'/checkpoint')
        torch.save(state, ospath+'/checkpoint/'+modelname+'.pth')
        best_acc = acc

if __name__=='__main__':
    # hyp
    lr=0.0001
    TMAX=200
    momentum=0.9
    startepoch=0
    endepoch=200
    best_acc = 0 
    RESUME=False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ospath=os.path.split(os.path.realpath(__file__))[0].replace("\\","/")
    print('Loading datas .......')
    transforms_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transforms_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    datapath =os.path.join(ospath,'data')
    train_data = CatDogDataset(file_path=datapath,flag='train',transform=transforms_train)
    print(train_data)
    test_data = CatDogDataset(file_path=datapath,flag='test',transform=transforms_test)
    print(test_data)


    train_loadata=DataLoader(
        train_data,batch_size=32,shuffle=True,num_workers=1
    )
    test_loadata = DataLoader(
        test_data,batch_size=2,shuffle=True,num_workers=1
    )
    print("data over.......")
    
    print('building model.....')
    net=Alexnet()
    net = net.to(device)
    modelname=type(net).__name__
    if RESUME:
        print('Resuming from checkpoint..... ')
        assert os.path.isdir(ospath+'/checkpoint'),'Error: no checkpoint find'
        checkpoint = torch.load(ospath+'/checkpoint/' + modelname +'.pth' )
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        startepoch = checkpoint['epoch']

    print('Prepare for tensorboard......')
    modelname=type(net).__name__ # get the model name
    if not os.path.exists(ospath+'/runs'):
        os.mkdir(ospath+'/runs')
    DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
    #time of we run the script
    TIME_NOW = datetime.now().strftime(DATE_FORMAT)
    writer = SummaryWriter(log_dir=os.path.join(
            ospath+'/runs',modelname,TIME_NOW))
    # draw the model structure in tensorboard
    td_inputtensor = torch.Tensor(1,3,224,224)
    if device == 'cuda' :
        td_inputtensor=td_inputtensor.cuda()
    writer.add_graph(net,td_inputtensor)
    print('tensorboard ready')


    print('loss, optimizer and scheduler......')
    loss = nn.CrossEntropyLoss()
    # optimizer = optim.adam(net.parameters(),lr)
    optimizer = optim.SGD(net.parameters(),lr=lr,momentum=momentum)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=TMAX)
    for epoch in range(startepoch ,endepoch):
        train(epoch)
        test(epoch)
        scheduler.step()
        
    writer.close()
