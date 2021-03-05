# coding:utf-8
import functools

from block import Block
import time
from transaction import Vout, Transaction
from account import get_account
from data.database import BlockChainDB, TransactionDB, UnTransactionDB, AccountDB
import torch
import ipfs.method as ipfs
import sys, os
import FedAvg.controller as fed
import torch.nn.functional as F
from collections import defaultdict
import numpy as np

"""
    目前是直接汇总，然后平均参数，得出准确率后上传
    由于保存模型的.pth无法解析
    所以还是要net.load()后平均
"""


def test_net(sum_parameters, model_id, total_iter, current_iter):
    sum_accu = 0
    num = 0
    net, testDataLoader, dev = fed.get_net()
    # 以下测试过程不计算梯度
    with torch.no_grad():
        net.load_state_dict(sum_parameters)
        for data, label in testDataLoader:
            data, label = data.to(dev), label.to(dev)
            preds = net(data)
            preds = torch.argmax(preds, dim=1)
            sum_accu += (preds == label).float().mean()
            num += 1
    # 保存参数和梯度，梯度信息置空
    grads = []
    state = {'net': net.state_dict(), 'grads': grads}
    torch.save(state,
               os.path.dirname(os.path.abspath(
                   __file__)) + "/modelData/fedModel/fedModel_id=" + str(model_id) + "total_iter=" + str(
                   total_iter) + "current_iter=" + str(current_iter))
    hash = ipfs.ipfs_add(os.path.dirname(os.path.abspath(
        __file__)) + "/modelData/fedModel/fedModel_id=" + str(model_id) + "total_iter=" + str(
        total_iter) + "current_iter=" + str(current_iter))
    return float(sum_accu / num), hash


def decide_which_method(timestamp):
    return timestamp % 4

def get_federated_model(txs, model_id, total_iter, current_iter):
    ipfs_hashes = []
    num = len(txs)
    sum_parameters = None
    users_grads = []
    parameters_arr = []
    row_para_arr = []
    i = 0
    result_para = None
    tx_arr = []
    for item in txs:
        ipfs_hashes.append(item['model_info']['model_ipfs_hash'])
    # for idx, usr in enumerate(ipfs_hashes):
    #     users_grads[idx, :] = usr.grads
    for item in ipfs_hashes:
        ipfs.ipfs_get(item, os.path.dirname(os.path.abspath(__file__)) + "/modelData/modelcache" + item + ".pth")
        # for cpu
        checkpoint = torch.load(
            os.path.dirname(os.path.abspath(__file__)) + "/modelData/modelcache" + item + ".pth", map_location='cpu')
        # for gpu
        # checkpoint = torch.load(os.path.dirname(os.path.abspath(__file__)) + "/modelData/modelcache" + item + ".pth")
        local_parameters = checkpoint['net']
        parameters_arr.append(local_parameters)
        # 得到para的一维序列
        net, testDataLoader, dev = fed.get_net()
        net.load_state_dict(local_parameters)
        row_para_arr.append(np.concatenate([i.data.numpy().flatten() for i in net.parameters()]))
        os.remove(os.path.dirname(os.path.abspath(__file__)) + "/modelData/modelcache" + item + ".pth")
        i = i + 1
    method_index = decide_which_method(txs[0]['timestamp'])
    print("防御方法:")
    if method_index == 0:
        print('no defend')
        result_para = mean_para(parameters_arr, num)
        tx_arr = txs
    elif method_index == 1:
        print('krum')

        index = krum(row_para_arr, num, corrupted_count=0)
        tx_arr = [txs[index]]
        result_para = parameters_arr[index]
    elif method_index == 2:
        print('trimmed')
        result_para = trimmed_mean(np.array(row_para_arr), num, corrupted_count=0)
        # trimmed 所有节点都贡献
        tx_arr = txs
    elif method_index == 3:
        print('bulyan')
        txs_index, result_para = bulyan(np.array(row_para_arr), num, corrupted_count=0)
        tx_arr = []
        for index in txs_index:
            tx_arr.append(txs[index])
    accuracy, hash = test_net(result_para, model_id, total_iter, current_iter)
    return hash, accuracy, tx_arr

def row_into_parameters(row, parameters):
    offset = 0
    for param in parameters:
        new_size = functools.reduce(lambda x,y:x*y, param.shape)
        current_data = row[offset:offset + new_size]

        param.data[:] = torch.from_numpy(current_data.reshape(param.shape))
        offset += new_size

def mean_para(parameters_arr, num):
    sum_parameters = None
    for single_para in parameters_arr:
        if sum_parameters is None:
            sum_parameters = {}
            for key, var in single_para.items():
                sum_parameters[key] = var.clone()
        else:
            for var in sum_parameters:
                sum_parameters[var] = sum_parameters[var] + single_para[var]
    for var in sum_parameters:
        sum_parameters[var] = (sum_parameters[var] / num)
    return sum_parameters


# krum是在k个模型中，选择一个与其他模型最为相似的,用梯度来作为选择
def _krum_create_distances(users_grads):
    distances = defaultdict(dict)
    for i in range(len(users_grads)):
        for j in range(i):
            distances[i][j] = distances[j][i] = np.linalg.norm(users_grads[i] - users_grads[j])
    return distances

# 改造krum算法，把从梯度中选择，改成从参数序列中选择
def krum(row_para_arr, users_count, corrupted_count, distances=None, return_index=True, debug=False):
    if not return_index:
        assert users_count >= 2 * corrupted_count + 1, (
        'users_count>=2*corrupted_count + 3', users_count, corrupted_count)
    non_malicious_count = users_count - corrupted_count
    minimal_error = 1e20
    minimal_error_index = -1

    if distances is None:
        distances = _krum_create_distances(row_para_arr)
    for user in distances.keys():
        errors = sorted(distances[user].values())
        current_error = sum(errors[:non_malicious_count])
        if current_error < minimal_error:
            minimal_error = current_error
            minimal_error_index = user
    if return_index:
        return minimal_error_index
    else:
        return row_para_arr[minimal_error_index]


# 针对梯度向量的每一列，去掉首位n个值，最后算出平均数
# 改成从参数向量的每一列
def trimmed_mean(row_para_arr, users_count, corrupted_count):
    number_to_consider = int(row_para_arr.shape[0] - corrupted_count) - 1
    current_paras = np.empty((row_para_arr.shape[1],), row_para_arr.dtype)

    for i, param_across_users in enumerate(row_para_arr.T):
        med = np.median(param_across_users)
        good_vals = sorted(param_across_users - med, key=lambda x: abs(x))[:number_to_consider]
        current_paras[i] = np.mean(good_vals) + med
    net, testDataLoader, dev = fed.get_net()
    row_into_parameters(current_paras,net.parameters())
    return net.state_dict()


# 先用krum循环找出users_count - 2*corrupted_count个，再用trmmied算法聚合
def bulyan(row_para_arr, users_count, corrupted_count):
    assert users_count >= 4*corrupted_count + 2
    set_size = users_count - 2*corrupted_count
    selection_set = []
    txs_index = []
    distances = _krum_create_distances(row_para_arr)
    while len(selection_set) < set_size:
        currently_selected = krum(row_para_arr, users_count - len(selection_set), corrupted_count, distances, True)
        selection_set.append(row_para_arr[currently_selected])
        txs_index.append(currently_selected)
        # remove the selected from next iterations:
        distances.pop(currently_selected)
        for remaining_user in distances.keys():
            distances[remaining_user].pop(currently_selected)
    return txs_index, trimmed_mean(np.array(selection_set), len(selection_set), 2*corrupted_count)

