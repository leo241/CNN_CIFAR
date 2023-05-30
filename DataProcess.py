from keras.utils import np_utils # 用于独热编码
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from copy import deepcopy

def cutout(img,label,size = 0.7):
    l = img.shape[0]
    h = img.shape[1]
    dh = int(h * size/2)
    dl = int(l * size/2)
    y = np.random.randint(h)
    x = np.random.randint(l)
    tmp = deepcopy(img)
    tmp[max(x - dl,0):min(x + dl,l),max(y - dh,0):min(y + dh,h)] = 0
    return tmp,label

def mixup(img1, img2, label1, label2):
    lam = np.random.uniform()
    mixed_img = lam * img1 + (1-lam) * img2
    mixed_img = mixed_img.astype('uint8')
    return mixed_img,(label1, label2, lam)


def cutmix(img1, img2, label1, label2, size = 0.7):
    l = img1.shape[0]
    h = img1.shape[1]
    dh = int(h * size / 2)
    dl = int(l * size / 2)
    y = np.random.randint(h)
    x = np.random.randint(l)
    tmp = deepcopy(img2)
    tmp[max(x - dl, 0):min(x + dl, l), max(y - dh, 0):min(y + dh, h),:] = img1[max(x - dl, 0):min(x + dl, l), max(y - dh, 0):min(y + dh, h),:]
    lam = (min(x + dl, l) - max(x - dl, 0)) * (min(y + dh, h) - max(y - dh, 0)) / h / l
    return tmp, (label1,label2,lam)


def get_data(istrain = True):
    train_dataset = datasets.cifar.CIFAR100(root='cifar100', train=istrain, transform=None,
                                            download=False)  # 如果是第一次运行把download换成True就可以了
    # test_dataset = datasets.cifar.CIFAR100(root='cifar100', train=False, transform=None, download=False)
    x = train_dataset.data
    y = train_dataset.targets
    train_list = list(zip(x,y))
    return train_list

def get_data_cutout(size = 0.5):
    train_dataset = datasets.cifar.CIFAR100(root='cifar100', train=True, transform=None,
                                            download=False)  # 如果是第一次运行把download换成True就可以了
    # test_dataset = datasets.cifar.CIFAR100(root='cifar100', train=False, transform=None, download=False)
    x = train_dataset.data
    y = train_dataset.targets
    train_list = list(zip(x,y))
    train_list2 = [cutout(item[0], item[1],size=size) for item in train_list]
    return train_list + train_list2

def get_data_mixup():
    train_dataset = datasets.cifar.CIFAR100(root='cifar100', train=True, transform=None,
                                            download=False)  # 如果是第一次运行把download换成True就可以了
    # test_dataset = datasets.cifar.CIFAR100(root='cifar100', train=False, transform=None, download=False)
    x = train_dataset.data
    y = train_dataset.targets
    train_list = list(zip(x,y))
    train_list2 = []
    for i in range(len(train_list)):
        id1,id2 = np.random.choice(len(train_list),2)
        train_list2.append(mixup(train_list[id1][0],train_list[id2][0],train_list[id1][1],train_list[id2][1]))
    return train_list + train_list2

def get_data_cutmix(size = 0.6):
    train_dataset = datasets.cifar.CIFAR100(root='cifar100', train=True, transform=None,
                                            download=False)  # 如果是第一次运行把download换成True就可以了
    # test_dataset = datasets.cifar.CIFAR100(root='cifar100', train=False, transform=None, download=False)
    x = train_dataset.data
    y = train_dataset.targets
    train_list = list(zip(x,y))
    train_list2 = []
    for i in range(len(train_list)):
        id1,id2 = np.random.choice(len(train_list),2)
        train_list2.append(cutmix(train_list[id1][0],train_list[id2][0],train_list[id1][1],train_list[id2][1],size=size))
    return train_list + train_list2

class CnnDataset(Dataset):  #
    '''
    继承了torch.utils.data.Dataset,用于加载数据，后续载入神经网络中
    '''

    def __init__(self, data,TensorTransform):
        self.data = data
        self.TensorTransform = TensorTransform


    def __getitem__(self, item,is_scaler = False):  # 这个是Dataset类的关键函数，形成数据的最终形式，通过迭代的形式喂给后续的神经网络
        X, Y = self.data[item]
        # print(X.dtype,X.shape)
        if is_scaler:
            x_scaler = []
            for i in range(3):
                x = X[:,:,i]
                scale_tmp = x - np.mean(x)/ np.std(x)
                scale_tmp = np.expand_dims(scale_tmp,2)
                x_scaler.append(scale_tmp)
            x_scaler = np.concatenate(x_scaler,axis=2)
            x_scaler = x_scaler.astype(np.float32) # float64 -> float32
            xoutput = self.TensorTransform(x_scaler)  # 变为张量
        else:
            xoutput = self.TensorTransform(X)
        # print(x_scaler.dtype,x_scaler.shape)
        if type(Y) == int:
            Y = np_utils.to_categorical(Y, 100) # 独热编码
        else: # 如果是存在mix的情况，像mixup cutmix
            y1,y2,p1 = Y
            p2 = 1 - p1
            Y = np_utils.to_categorical(y1, 100) * p1 + np_utils.to_categorical(y1, 100) * p2
        return xoutput, torch.tensor(Y)

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    train_list = get_data()
    img1,l1 = train_list[0]
    img2, l2 = train_list[1]
    img3, l3 = train_list[2]
    plt.imshow(img1)
    plt.imshow(img2)
    plt.imshow(img3)
    img1_c = cutout(img1,l1,size=0.65)[0]
    img2_c = cutout(img2, l1, size=0.65)[0]
    img3_c = cutout(img3, l1, size=0.65)[0]
    plt.imshow(img1_c)
    plt.imshow(img2_c)
    plt.imshow(img3_c)
    img1_m = mixup(img1,img2, l1,l2)
    img2_m = mixup(img2,img3, l2,l3)
    img3_m = mixup(img3,img1, l3, l1)
    plt.imshow(img1_m[0])
    plt.imshow(img2_m[0])
    plt.imshow(img3_m[0])
    img1_cm = cutmix(img1, img2, l1, l2)
    img2_cm = cutmix(img2, img3, l2, l3)
    img3_cm = cutmix(img3, img1, l3, l1)
    plt.imshow(img1_cm[0])
    plt.imshow(img2_cm[0])
    plt.imshow(img3_cm[0])
