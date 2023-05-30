from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from predict import cnn
from keras.utils import np_utils # 用于独热编码

def predict_all(net,test_dataset = False,train = False,length = 2500,loss_func = nn.CrossEntropyLoss()): # 给定验证集列表，得到预测的结果
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 使用gpu加速
    net = net.to(device)
    if not test_dataset:
        test_dataset = datasets.cifar.CIFAR100(root='cifar100', train=train, transform=None, download=False)
    x = test_dataset.data
    y = test_dataset.targets
    y_array = np_utils.to_categorical(y)
    whole_length = len(x) # 整个数据集的长度，如果是train就是60000，如果是test就是10000
    id = 0
    cnt = 0
    ylosss = []
    while id < whole_length:
        x1,y1 = x[id:id + length],y[id:id + length]
        y_array1 = y_array[id:id + length, :]
        ytorch = torch.tensor(y_array1).to(device)
        TensorTransform = transforms.Compose([  # transform to figure, for further passing to nn
            transforms.ToTensor(),  # ToTensor会给灰度图像自动增添一个维度
        ])
        vx = [TensorTransform(item).unsqueeze(0) for item in x1]  # 256,256 -> 1，1，256,256

        vx = torch.cat(vx, dim=0).to(device) # 这个位置记得加入GPU！一切预测的地方都要注意
        net.train()
        with torch.no_grad():
            ypre = net(vx)
            loss = float(loss_func(ypre, ytorch))
            ylosss.append(loss)
            ypre = torch.argmax(ypre,dim=1)
            cnt += sum(np.asarray(y1) == np.asarray(ypre.cpu()))
        id += length # 滑动窗口
    accuracy = cnt / whole_length
    loss = sum(ylosss)/ len(ylosss)
    return round(accuracy,3),round(loss, 3)

if __name__ == '__main__':
    length = 2500  # 样本的截断长度,正常应设置成测试集的大小--10000
    model_root = 'model_save'
    train_writer = SummaryWriter('logs_train')
    test_writer = SummaryWriter('logs_test')

    print('常规预测')
    print('截断样本量：', length)
    print('-------------------------------------')

    net = cnn(100)
    models = list(range(470000, 475001, 1000))
    for model in tqdm(models):
        net.load_state_dict(torch.load(f'{model_root}/{model}.pth'))
        train_accuracy,train_loss = predict_all(net, train=True, length=length)
        test_accuracy,test_loss = predict_all(net, train=False, length=length)
        train_writer.add_scalar('train acc',train_accuracy,int(model/1000))
        train_writer.add_scalar('train loss', train_loss, int(model / 1000))
        test_writer.add_scalar('test acc', test_accuracy, int(model / 1000))
        test_writer.add_scalar('test loss', test_loss, int(model / 1000))
        line = [int(model / 100), train_accuracy, train_loss, test_accuracy, test_loss]
        line = [str(i) for i in line]
        line = '\t'.join(line) + '\n'
        with open('logs.txt','a') as file:
            file.write(line)
            file.close()
    train_writer.close()
    test_writer.close()
