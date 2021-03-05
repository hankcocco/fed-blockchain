import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from Models import Mnist_2NN, Mnist_CNN
from clients import ClientsGroup, client
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nc', '--num_of_clients', type=int, default=5, help='numer of the clients')
parser.add_argument('-cf', '--cfraction', type=float, default=0.1, help='C fraction, 0 means 1 client, 1 means total clients')
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='mnist_2nn', help='the global_models to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-vf', "--val_freq", type=int, default=5, help="global_models validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global global_models save frequency(of communication)')
parser.add_argument('-ncomm', '--num_comm', type=int, default=1000, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')


def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


if __name__=="__main__":
    args = parser.parse_args()
    args = args.__dict__

    test_mkdir(args['save_path'])

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    net = None
    if args['model_name'] == 'mnist_2nn':
        net = Mnist_2NN()
    elif args['model_name'] == 'mnist_cnn':
        net = Mnist_CNN()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)
    net = net.to(dev)

    loss_func = F.cross_entropy
    opti = optim.SGD(net.parameters(), lr=args['learning_rate'])

    myClients = ClientsGroup('mnist', args['IID'], args['num_of_clients'], dev)
    # 获取testDataLoader
    testDataLoader = myClients.test_data_loader

    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))

    global_parameters = {}
    init_flag = 1
    # 初始化网络参数
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

    torch.save(net.state_dict(), 'global_models/server_iter_0.pth')
    global_parameters_path = 'global_models/server_iter_0.pth'
    # 迭代轮次
    for i in range(args['num_comm']):
        print("communicate round {}".format(i+1))

        # order = np.random.permutation(args['num_of_clients'])
        # 随机打乱的clients序列
        # clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]
        clients_in_comm = ['client{}'.format(i) for i in range(num_in_comm)]
        sum_parameters = None
        # tqdm是进度条插件，clients分别训练，合并global参数
        for client in tqdm(clients_in_comm):
            # 第一次迭代时，输出client自己的准确率
            # 从文件读取参数
            local_parameters_path = myClients.clients_set[client].localUpdate(args['epoch'], args['batchsize'], i, net, loss_func, opti, global_parameters_path, init_flag)
            local_parameters = torch.load(local_parameters_path)
            if sum_parameters is None:
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = var.clone()
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + local_parameters[var]

        init_flag = 0
        # 模型参数取平均
        for var in global_parameters:
            global_parameters[var] = (sum_parameters[var] / num_in_comm)
        # 存储平均后的global models
        torch.save(global_parameters_path, './global_models/server_iter'+str(i))
        global_parameters_path = './global_models/server_iter'+str(i)
        # 按频率输出正确率
        with torch.no_grad():
            if (i + 1) % args['val_freq'] == 0:
                net.load_state_dict(global_parameters, strict=True)
                sum_accu = 0
                num = 0
                for data, label in testDataLoader:
                    data, label = data.to(dev), label.to(dev)
                    preds = net(data)
                    preds = torch.argmax(preds, dim=1)
                    sum_accu += (preds == label).float().mean()
                    num += 1
                print('accuracy: {}'.format(sum_accu / num))
                # 画图
                writer1 = SummaryWriter('runs/federate_accuracy_every_iterator')
                writer1.add_scalar('accuracy', sum_accu / num, global_step=i)
        # 保存文件
        if (i + 1) % args['save_freq'] == 0:
            torch.save(net, os.path.join(args['save_path'],
                                         '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['model_name'],
                                                                                                i, args['epoch'],
                                                                                                args['batchsize'],
                                                                                                args['learning_rate'],
                                                                                                args['num_of_clients'],
                                                                                                args['cfraction'])))

