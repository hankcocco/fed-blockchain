
import os
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Subset
import torch
import torch.nn.functional as F
from torch import optim
import sys, os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import ipfs.method as ipfs
from Models import Mnist_2NN, Mnist_CNN
from clients import client
from getData import GetDataSet
from data.database import StartDB


from torch.utils.data import DataLoader

GPU = "0"
EPOCH = 5
BATCHSIZE = 10
MODEL_NAME = 'mnist_2nn'
LEARNING_RATE = 0.01
IDD = 0
CID = 1




def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def init_net():
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # 单纯cpu
    # dev = torch.device("cpu")
    net = None
    if MODEL_NAME == 'mnist_2nn':
        net = Mnist_2NN()
    elif MODEL_NAME == 'mnist_cnn':
        net = Mnist_CNN()
    # for gpu,
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)
    net = net.to(dev)

    loss_func = F.cross_entropy
    opti = optim.SGD(net.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
    return net, dev, loss_func, opti


def get_train_set(start, size):
    mnistDataSet = GetDataSet('mnist', IDD)
    train_data = mnistDataSet.train_set
    train_data = Subset(train_data, range(start, start+size))
    return train_data


"""
    目前默认Minist前2000数据是测试数据
"""


def get_test_set():
    mnistDataSet = GetDataSet('mnist', IDD)
    test_data = mnistDataSet.test_set
    train_data = Subset(test_data, range(0, 3000))
    return train_data


"""
    开始新一轮联邦学习时，初始化网络
"""


def new_model_init():
    net, dev, loss_func, opti = init_net()
    # 保存参数
    # torch.save(net.state_dict(),  os.path.dirname(os.path.abspath(__file__))+'/local_models/init_model.pth')
    # 保存模型,初始化梯度信息为空
    grads = []
    state = {'net': net.state_dict(), 'grads': grads}
    torch.save(state, os.path.dirname(os.path.abspath(__file__)) + '/local_models/init_model.pth')
    local_parameters_path = '/local_models/init_model.pth'
    ab_path = os.path.dirname(os.path.abspath(__file__)) + local_parameters_path
    ipfs_hash = ipfs.ipfs_add(ab_path)
    test_set = get_test_set()
    sum_accu = 0
    num = 0
    # test_set = torchvision.datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor(),
    #                                               download=False)
    # test_set = Subset(test_set, range(0, 2000))
    testDataLoader = DataLoader(dataset=test_set, batch_size=100, shuffle=True)
    for data, label in testDataLoader:
        data, label = data.to(dev), label.to(dev)
        preds = net(data)
        preds = torch.argmax(preds, dim=1)
        sum_accu += (preds == label).float().mean()
        num += 1
    return ipfs_hash, float(sum_accu / num)

def get_net():
    net, dev, loss_func, opti = init_net()
    test_set = get_test_set()
    testDataLoader = DataLoader(dataset=test_set, batch_size=100, shuffle=True)
    return net, testDataLoader, dev

def train(model_id, iter, global_ipfs):
    # args = parser.parse_args()
    # args = args.__dict__
    # test_mkdir()
    net, dev, loss_func, opti = init_net()
    sdb = StartDB()
    start, end = sdb.get()
    train_set = get_train_set(start, end)
    test_set = get_test_set()
    local_parameter_hash, accuracy = client(train_set, CID, dev).localUpdate(EPOCH, BATCHSIZE,
                                                                                     iter, net, loss_func, opti,
                                                                                     global_ipfs, test_set)

    print('上传训练模型，id '+str(model_id)+" 轮次: "+str(iter)+" 准确率: "+str(accuracy))

    return model_id, local_parameter_hash, accuracy, iter


# if __name__ == "__main__":
#     # import sys
#     # BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     # sys.path.append(BASE_DIR)
#     new_model_init()
