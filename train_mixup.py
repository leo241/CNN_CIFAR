# 锁住卷积层的参数不变，只采用较大的L2正则化去训练训练线性层
'''
greatest artests always hide themselves in their work
author:刘津铭
ds : 20230527
email: 22210980054@m.fudan.edu.cn
feel free to copy my code ,because I always do the same thing in terms of deep learning :)
'''
from DataProcess import get_data, CnnDataset,get_data_cutout,get_data_mixup,get_data_cutmix# 引入本地文件DataProcess.py

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from PIL import Image
import numpy as np
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")# 使用gpu加速


DROP_PRO = 0.2 # 定义随机森林的丢弃概率


class cnn(nn.Module): # construction of netral network
    def __init__(self,num_classes):
        super(cnn, self).__init__()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d( # 1 224 224
                in_channels=3, # input rgb size
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2 # （l - k + 2p）/s + 1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(DROP_PRO),
            nn.Conv2d(  # 1 224 224
                in_channels=16,  # input rgb size
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2  # （l - k + 2p）/s + 1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 16,128,128
        )
        # 16 224 224
        self.conv2 = nn.Sequential(
            nn.Conv2d( # 1 224 224
                in_channels=32, # input rgb size
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2 # （l - k + 2p）/s + 1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(DROP_PRO),
            nn.Conv2d(  # 1 224 224
                in_channels=64,  # input rgb size
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2  # （l - k + 2p）/s + 1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 16,128,128
        )
        #
        self.conv3 = nn.Sequential(
            nn.Conv2d( # 1 224 224
                in_channels=128, # input rgb size
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=2 # （l - k + 2p）/s + 1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(DROP_PRO),
            nn.Conv2d(  # 1 224 224
                in_channels=256,  # input rgb size
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=2  # （l - k + 2p）/s + 1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 16,128,128
        )

        self.fc1 = nn.Linear(25088, 2048)
        self.out= nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv3(x)
        x = x.view(x.shape[0], -1) # flatten
        # print(x.size(), '进入全连接层前的维度')
        x = self.relu(self.fc1(x))
        x = self.out(x) # 全连接层也需要激活函数，切记
        # x = self.fc1(x)
        # x = self.out(x)
        # x = self.softmax(x)
        return x


class nn_processor:
    def __init__(self, train_loader, valid_loader=None, valid_list=None):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.valid_list = valid_list

    def train(self, net, lr=0.01, EPOCH=40, max_iter=500, save_iter=500, plot_iter=2000, first_iter=0,print_iter = 100,
              loss_func=nn.BCEWithLogitsLoss(), save_path = False,test_dataset = False,roll = True,weight_decay = 0,lr_beta = 1):
        # train_writer = SummaryWriter('logs_train')
        # test_writer = SummaryWriter('logs_test')
        optimizer = torch.optim.Adam(net.parameters(), lr=lr,weight_decay=weight_decay)  # 这里没有加入encoder.parameters(),所以自编码网络的参数不会改变
        net = net.to(device)
        i = 0
        loss_train_list = list()
        loss_valid_list = list()
        iter_list = list()
        stop = False
        for epoch in tqdm(range(EPOCH)):
            # if epoch % 2 == 0: # 学习率随epoch衰减策略
            #     for p in optimizer.param_groups:
            #         p['lr'] = lr_beta
            if stop == True:
                break
            print(f'\nepoch {epoch}')
            for step, (x, y) in enumerate(self.train_loader):
                x, y = x.to(device), y.to(device)
                y_pre = net(x)
                # output1 = output1.to(torch.float)
                # y = y.to(torch.float)
                # output2 = output2.to(torch.float)
                # y2 =y2.to(torch.float)
                loss = loss_func(y_pre, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                i += 1

                if i % print_iter == 0:
                    print(f'\nepoch:{epoch+1}\niteration: {i+first_iter}')
                    print('train loss:', float(loss))
                    for k,(xv,yv) in enumerate(self.valid_loader):
                        xv, yv = xv.to(device), yv.to(device)
                        y_prev = net(xv)
                        lossv = loss_func(y_prev, yv)
                        print('valid loss:', float(lossv))
                        break
                    # train_writer.add_scalar('loss', loss, i)
                    # test_writer.add_scalar('loss', lossv, i)
                    if test_dataset:
                        accuracy = predict(net, test_dataset)
                    if i >= max_iter:  # 达到最大迭代，保存模型
                        stop = True
                        torch.save(net.state_dict(), f'{save_path}/{i+first_iter}.pth')
                        print('model saved!')
                        break
                    if save_path: # 如果指定了保存模型和路径
                        if i % save_iter == 0:  # 临时保存
                            if i != save_iter and i % 50000 != 0 and roll == True:
                                os.remove(f'{save_path}/{i+first_iter-save_iter}.pth')
                            torch.save(net.state_dict(), f'{save_path}/{i+first_iter}.pth')
                            print(f'model temp {i+first_iter} saved!')
                    #
                    # print('val ac:', DataProcess().predict(net, valid_list))

if __name__ == '__main__':
    batch_size = 256 # 扩大了batch_size 32 -> 256
    first_iter = 472000 # 注：在batch_size = 16下，一个epoch约等于3000个iteration
    print_iter = 100 # 打印频次缩小10倍
    save_iter = 500 # 保存频次缩小10倍
    max_iter = 10000 # 只看5000 iter
    Epoch = 20000
    lr = 0.0001 # CNN_dropout 前20万个iteration：0.002
    # ,从20万开始降低10倍为0.0002
    # ，25万开始降为0.00002,30万开始继续降10倍
    # ，35万后回调至0.00002
    # ，44万加至0.00003
    # ，46万回调0.00001
    lr_beta = 1 # lr 衰减策略
    # L2 = 0.005 # L2正则化,用于应对CFAR-100这种逆天过拟合数据集
    # L2 = 0
    L2 = 0.0075 # 前20万采用0.00001， 之后扩大十倍 -》 0.0001 30万后再扩大10倍 0.001 43万扩大到0.01,44万又改回了0.0075
    save_path = 'model_save' # 保存模型的文件夹，如需保存模型请注释掉下一行
    loss_func = nn.CrossEntropyLoss()
    # save_path = False # 不进行任何保存操作

    # train_list = get_data()
    train_list = get_data_mixup()
    # train_list = get_data_cutmix()
    valid_list = get_data(istrain = False)
    TensorTransform = transforms.Compose([  # transform to figure, for further passing to nn
        transforms.ToTensor(),  # ToTensor会给灰度图像自动增添一个维度
    ])
    train_data = CnnDataset(train_list, TensorTransform=TensorTransform)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, # 这里因为cifar100数据集自带随机性，所以不用随机打乱
                              num_workers=0)  # batch_size是从这里的DataLoader传递进去的
    valid_data = CnnDataset(valid_list, TensorTransform=TensorTransform)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True,  # 这里因为cifar100数据集自带随机性，所以不用随机打乱
                              num_workers=0)  # batch_size是从这里的DataLoader传递进去的
    net = cnn(100) # 新建一个CNN二分类网络，要保证每一折上的交叉验证都是从0开始训练
    if first_iter != 0:
        net.load_state_dict(torch.load(f'{save_path}/{first_iter}.pth'))
    models = net.modules()
    net.train()  # 开启dropout模式
    for p in models:
        if p._get_name() != 'Linear':
            p.requires_grad = False # 将所有卷积层的参数固定

    cnn_processor = nn_processor(train_loader,valid_loader=valid_loader)
    test_dataset = datasets.cifar.CIFAR100(root='cifar100', train=False, transform=None, download=False)
    cnn_processor.train(net,lr=lr, EPOCH=Epoch, max_iter=max_iter,
                        print_iter=print_iter,save_path= save_path,first_iter=first_iter,loss_func=loss_func,
                        test_dataset=False,roll=False,save_iter=save_iter,weight_decay=L2,lr_beta=lr_beta)
    # torch.save(net.state_dict(), f'{save_path}/final.pth')

