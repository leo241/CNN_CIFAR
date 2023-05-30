import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torch
import torch.nn as nn
from tqdm import tqdm

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
            nn.Dropout(0),
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
            nn.Dropout(0),
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
            nn.Dropout(0),
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

def predict_all(net,test_dataset = False,train = False,length = 2500): # 给定验证集列表，得到预测的结果
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 使用gpu加速
    net = net.to(device)
    if not test_dataset:
        test_dataset = datasets.cifar.CIFAR100(root='cifar100', train=train, transform=None, download=False)
    x = test_dataset.data
    y = test_dataset.targets
    whole_length = len(x) # 整个数据集的长度，如果是train就是60000，如果是test就是10000
    id = 0
    cnt = 0
    while id < whole_length:
        x1,y1 = x[id:id + length],y[id:id + length]
        TensorTransform = transforms.Compose([  # transform to figure, for further passing to nn
            transforms.ToTensor(),  # ToTensor会给灰度图像自动增添一个维度
        ])
        vx = [TensorTransform(item).unsqueeze(0) for item in x1]  # 256,256 -> 1，1，256,256

        vx = torch.cat(vx, dim=0).to(device) # 这个位置记得加入GPU！一切预测的地方都要注意
        net.train()
        with torch.no_grad():
            ypre = net(vx)
            ypre = torch.argmax(ypre,dim=1)
            cnt += sum(np.asarray(y1) == np.asarray(ypre.cpu()))
        id += length # 滑动窗口
    accuracy = cnt / whole_length
    return round(accuracy,3)


# def predict(net,test_dataset = False,train = False,length = 10000): # 给定验证集列表，得到预测的结果
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 使用gpu加速
#     net = net.to(device)
#     if not test_dataset:
#         test_dataset = datasets.cifar.CIFAR100(root='cifar100', train=train, transform=None, download=False)
#     x = test_dataset.data
#     y = test_dataset.targets
#     x,y = x[0:length],y[0:length]
#     TensorTransform = transforms.Compose([  # transform to figure, for further passing to nn
#         transforms.ToTensor(),  # ToTensor会给灰度图像自动增添一个维度
#     ])
#     vx = [TensorTransform(item).unsqueeze(0) for item in x]  # 256,256 -> 1，1，256,256
#
#     vx = torch.cat(vx, dim=0).to(device) # 这个位置记得加入GPU！一切预测的地方都要注意
#     # if cut_dropout:
#     #     net.eval() # 关闭dropout模式
#     # else:
#     net.train()
#     with torch.no_grad():
#         ypre = net(vx)
#         ypre = torch.argmax(ypre,dim=1)
#         accuracy = sum(np.asarray(y) == np.asarray(ypre.cpu()))/len(y)
#     return round(accuracy,3)


# def predict_by_sample(net,test_dataset = False,train = False,length = 10000,is_tqdm = True): # 逐个验证,防止大量验证导致cpu不足
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 使用gpu加速
#     net = net.to(device)
#     # if cut_dropout:
#     #     net.eval() # 关闭dropout模式
#     # else:
#     net.train()
#     if not test_dataset:
#         test_dataset = datasets.cifar.CIFAR100(root='cifar100', train=train, transform=None, download=False)
#     x = test_dataset.data
#     y = test_dataset.targets
#
#     TensorTransform = transforms.Compose([  # transform to figure, for further passing to nn
#         transforms.ToTensor(),  # ToTensor会给灰度图像自动增添一个维度
#     ])
#     cnt = 0
#     if is_tqdm:
#         for i in tqdm(range(length)):
#             x1, y1 = x[i], y[i]
#             vx = TensorTransform(x1).unsqueeze(0)  # 256,256 -> 1，1，256,256
#             vx = vx.to(device) # 这个位置记得加入GPU！一切预测的地方都要注意
#             with torch.no_grad():
#                 ypre = net(vx)
#                 ypre = torch.argmax(ypre,dim=1)
#                 if int(ypre.cpu()) == y1:
#                     cnt += 1
#     else:
#         for i in range(length):
#             x1, y1 = x[i], y[i]
#             vx = TensorTransform(x1).unsqueeze(0)  # 256,256 -> 1，1，256,256
#             vx = vx.to(device) # 这个位置记得加入GPU！一切预测的地方都要注意
#             with torch.no_grad():
#                 ypre = net(vx)
#                 ypre = torch.argmax(ypre,dim=1)
#                 if int(ypre.cpu()) == y1:
#                     cnt += 1
#     accuracy = cnt/length
#     return round(accuracy,3)

if __name__ == '__main__':
    length = 20 # 切片长度，防止一次性cpu装不下
    model_root = 'model_save'

    print('常规预测')
    print('截断样本量：',length)
    print('-------------------------------------')

    net = cnn(100)  # 新建一个CNN二分类网络，要保证每一折上的交叉验证都是从0开始训练
    models = list(range(472000,482001,1000))
    models = [472000]
    for model in models:
        net.load_state_dict(torch.load(f'{model_root}/{model}.pth'))
        print(model, '训练集准确率:', predict_all(net, train=True, length=length))
        print(model,'测试集准确率:',predict_all(net,train=False,length=length))
        # train_writer = SummaryWriter('logs_train')
        # test_writer = SummaryWriter('logs_test')
        # print(model,'train:',predict(net,train=True,length=length),'test:',predict(net, train=False,length=length)) # 快速预测 但是大样本cpu容易炸
