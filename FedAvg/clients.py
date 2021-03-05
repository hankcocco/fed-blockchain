import torch
from torch.utils.data import DataLoader
from getData import GetDataSet
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import sys, os
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import ipfs.method as ipfs



class client(object):
    def __init__(self, trainDataSet, clientNo, dev):
        self.train_ds = trainDataSet
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None
        self.clientNo = clientNo

    def localUpdate(self, localEpoch, localBatchSize, iter, Net, lossFun, opti, global_ipfs, test_set):
        print(self.dev)
        global_path = ipfs.ipfs_get(global_ipfs, os.path.dirname(os.path.abspath(__file__))+'/global_models/'+global_ipfs+'.pth')
        # 加载参数
        # Net.load_state_dict(torch.load(global_path), strict=True)
        # 改为从文件直接加载整个网络
        # Net = torch.load(global_path)
        # for cpu
        # checkpoint = torch.load(global_path, map_location='cpu')
        # for GPU
        checkpoint = torch.load(global_path)
        Net.load_state_dict(checkpoint['net'])
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True,)
        print("开始训练")
        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                # 先清零再训练，获取到梯度
                opti.zero_grad()
                preds = Net(data)
                loss = lossFun(preds, label)
                # 反向传播求梯度
                loss.backward()
                # 更新权重参数
                opti.step()
        # 保存梯度信息
        print("训练完毕")
        grads = np.concatenate([param.grad.data.cpu().numpy().flatten() for param in Net.parameters()])
        state = {'net': Net.state_dict(), 'grads': grads}
        # torch.save(Net.state_dict(), os.path.dirname(os.path.abspath(__file__))+'/local_models/client'+str(self.clientNo)+"_iter"+str(iter))
        # 存储参数及梯度,上传ipfs
        torch.save(state, os.path.dirname(os.path.abspath(__file__))+'/local_models/client'+str(self.clientNo)+"_iter"+str(iter))
        local_parameters_path = '/local_models/client'+str(self.clientNo)+"_iter"+str(iter)
        ab_path = os.path.dirname(os.path.abspath(__file__))+local_parameters_path
        ipfs_hash = ipfs.ipfs_add(ab_path)
        print("上传完毕")
        # 测试，返回准确率
        sum_accu = 0
        num = 0
        # test_set = torchvision.datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor(),
        #                                               download=False)
        # test_set = Subset(test_set, range(0, 2000))
        testDataLoader = DataLoader(dataset=test_set, batch_size=100, shuffle=False)
        for data, label in testDataLoader:
            data, label = data.to(self.dev), label.to(self.dev)
            preds = Net(data)
            preds = torch.argmax(preds, dim=1)
            sum_accu += (preds == label).float().mean()
            num += 1
            # print('accuracy of client ' + str(self.clientNo) + ': {}'.format(sum_accu / num))
        # 删除暂存模型
        # os.remove(os.path.dirname(os.path.dirname(os.path.abspath(__file__))+'/global_models/'+global_ipfs+'.pth'))
        # os.remove(os.path.dirname(os.path.abspath(__file__)) + '/local_models/client' + str(self.clientNo) + "_iter" + str(iter))
        return ipfs_hash, float(sum_accu/num)

    def local_val(self):
        pass


"""
    以下是分配clients集群，测试用
"""
#
# class ClientsGroup(object):
#     def __init__(self, dataSetName, isIID, numOfClients, dev):
#         self.data_set_name = dataSetName
#         self.is_iid = isIID
#         self.num_of_clients = numOfClients
#         self.dev = dev
#         self.clients_set = {}
#
#         self.test_data_loader = None
#
#         self.dataSetBalanceAllocation()
#
#     def dataSetBalanceAllocation(self):
#         mnistDataSet = GetDataSet(self.data_set_name, self.is_iid)
#         # 修改,测试集固定2000：
#         test_data = torchvision.datasets.MNIST(root=os.path.dirname(os.path.abspath(__file__))+"/data", train=False, transform=transforms.ToTensor(),
#                                                download=False)
#         test_data = Subset(test_data, range(0, 2000))
#         #
#         # 原代码：
#         # test_label = torch.argmax(torch.tensor(mnistDataSet.test_label), dim=1)
#         # self.test_data_loader = DataLoader(TensorDataset( test_data, test_label), batch_size=100, shuffle=False)
#         # train_data = mnistDataSet.train_data
#         # 修改：
#         self.test_data_loader = DataLoader(dataset=test_data, batch_size=100, shuffle=False)
#         train_data = mnistDataSet.train_set
#         # 原代码：
#         # train_label = mnistDataSet.train_label
#         # shard_size = mnistDataSet.train_data_size // self.num_of_clients // 2
#         # shards_id = np.random.permutation(mnistDataSet.train_data_size // shard_size)
#         print(len(train_data))
#         # shard_size = mnistDataSet.train_data_size // self.num_of_clients
#         shard_size = 500
#         # 分配数据集
#         for i in range(self.num_of_clients):
#             # shards_id1 = shards_id[i * 2]
#             # shards_id2 = shards_id[i * 2 + 1]
#             # data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
#             # data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
#             # label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
#             # label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
#             # local_data, local_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
#             # local_label = np.argmax(local_label, axis=1)
#             # someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
#             # self.clients_set['client{}'.format(i)] = someone
#             local_data_set = Subset(train_data, range(i * shard_size, (i + 1) * shard_size))
#             print(len(local_data_set))
#             someone = client(local_data_set, i, self.dev)
#             self.clients_set['client{}'.format(i)] = someone

