"""
挖矿，包含丢弃小链、创世块
创建新的训练任务（模型初始块）
"""
# coding:utf-8
from block import Block
import time
from transaction import Vout, Transaction
from account import get_account
from data.database import BlockChainDB, TransactionDB, UnTransactionDB, AccountDB
import consensus
from FedAvg.controller import train as NewModelEvent
MAX_COIN = 21000000
REWARD = 20
# 每轮迭代 最小收到的节点模型数量
MIN_MODEL_NUM = 3
"""
没有验证账户功能
"""


def reward(default_account):
    reward = Vout(get_account()['address'], REWARD)
    tx = Transaction([], reward, AccountDB().find_some(default_account))
    return tx


def coinbase(default_account):
    """
    First block generate.
    """
    rw = reward(default_account)
    cb = Block(0, int(time.time()), [rw.hash], "")
    nouce = cb.pow()
    cb.make(nouce)
    # Save block and transactions to database.
    BlockChainDB().insert(cb.to_dict())
    TransactionDB().insert(rw.to_dict())
    return cb


def get_all_untransactions():
    UnTransactionDB().all_hashes()


"""
    清除过时的Model_update交易
"""


def clear_out_time_tx():
    last_block = BlockChainDB().last()
    if last_block['model_info'] != {}:
        last_info = {'model_id': last_block['model_info']['model_id'],
                     'current_iter': last_block['model_info']['current_iter'],
                     'total_iter': last_block['model_info']['total_iter']
                     }
        untxdb = UnTransactionDB()
        for item in untxdb.find_is_model():
            if item['model_info']['model_id'] != last_info['model_id'] or item['model_info']['iter_num'] != last_info['current_iter'] + 1:
                UnTransactionDB().delete(item)
                # 如果达到了预定的迭代轮次，不再迭代，删除（一个冗余判定，传播的时候已经判定不会传播了）
            if item['model_info']['iter_num'] > last_info['total_iter']:
                UnTransactionDB().delete(item)


"""
    当untx中的模型更新达到一定数量(min_model_num)后，将它们包含到区块，并计算出global model后上传
    目前只是pow证明后即可记账
    初步设想，当一个节点记账时，需要从前一个区块开始，逐个验证所有global model组合正确情况，才能上链
"""


def mine_upload_models(default_account=0):
    from_addr = AccountDB().find_some(0)
    last_block = BlockChainDB().last()
    untxdb = UnTransactionDB()
    model_num = len(untxdb.find_is_model())
    model_info = last_block['model_info']
    # 这里还需要增加一个时间戳判定
    if model_num >= MIN_MODEL_NUM:
        new_model_hash, accuracy, untxs = consensus.get_federated_model(untxdb.find_is_model(), model_info['model_id'], model_info['total_iter'],  model_info['current_iter']+1)
        model_info['init_ipfs_hash'] = new_model_hash
        model_info['current_iter'] = model_info['current_iter'] + 1
        model_info['accuracy'] = accuracy
        # 成块
        untx_hashes = []
        for item in untxs:
            untx_hashes.append(item['hash'])
            UnTransactionDB().delete(item)

        cb = Block(last_block['index'] + 1, int(time.time()), untx_hashes, last_block['hash'], 1, model_info)
        nouce = cb.pow()
        cb.make(nouce)
        # 验证一遍last_block 避免重复挖矿（之前另外的节点已经提交）
        last_block = BlockChainDB().last()
        print(last_block['index'])
        print(cb.to_dict()['index'])
        if last_block['index'] != cb.to_dict()['index']:
            # Save block and transactions to database.
            BlockChainDB().insert(cb.to_dict())
            TransactionDB().insert(untxs)
            # Broadcast to other nodes
            Block.spread(cb.to_dict(), untxs, from_addr)
            # Transaction.blocked_spread(untxs)
            # 奖励
            # rw = reward(default_account)
            # untxs = untxdb.find_not_model()
            # untxs.append(rw.to_dict())
            # Transaction.blocked_spread(untxs)
            return cb
    return 0


def mine(default_account=0):
    """
    Main miner method.
    """
    # Found last block and unchecked transactions.
    from_addr = AccountDB().find_some(0)
    last_block = BlockChainDB().last()
    if len(last_block) == 0:
        last_block = coinbase(default_account)
        Block.spread(last_block.to_dict(), [], from_addr)
        return last_block
    else:
        # 继承上一个区块的model_info
        model_info = last_block['model_info']
        clear_out_time_tx()
    untxdb = UnTransactionDB()
    # 有未交易普通信息，才Mine
    if untxdb.find_not_model():
        # Miner reward
        rw = reward(default_account)
        untxs = untxdb.find_not_model()
        untxs.append(rw.to_dict())
        # untxs_dict = [untx.to_dict() for untx in untxs]
        untx_hashes = untxdb.not_model_hashes()
        # 只清空Mine了的交易
        untx_is_model = untxdb.find_is_model()
        untxdb.clear()
        for item in untx_is_model:
            untxdb.write(item)
        # Miner reward is the first transaction.
        untx_hashes.insert(0, rw.hash)
        # 如果是普通区块，直接继承上一个区块的Model_info
        cb = Block(last_block['index'] + 1, int(time.time()), untx_hashes, last_block['hash'], 0, model_info)
        nouce = cb.pow()
        cb.make(nouce)
        # 验证一遍last_block 避免重复挖矿（之前另外的节点已经提交）
        last_block = BlockChainDB().last()
        if last_block['index'] != cb.to_dict()['index']:
            # Save block and transactions to database.
            BlockChainDB().insert(cb.to_dict())
            TransactionDB().insert(untxs)
            # Broadcast to other nodes
            Block.spread(cb.to_dict(), untxs, from_addr)
            # Transaction.blocked_spread(untxs)
            return cb
    elif untxdb.find_is_model():
        return mine_upload_models()
    else:
        return 0

def valid_before_insert():
    if len(BlockChainDB().find_is_model()) != 0:
        last = BlockChainDB().find_is_model()[-1]
        # 需要上一个模型已经训练完
        if last['model_info']['current_iter'] != last['model_info']['total_iter']:
            return False
    return True


# 创建一个新训练任务
def init_new_model_block(model_id, init_ipfs_hash, total_iter, accuracy, current_iter=0):
    is_model = 1
    from_addr = AccountDB().find_some(0)
    model_info = {'model_id': model_id, 'init_ipfs_hash': init_ipfs_hash, 'total_iter': total_iter,
                  'current_iter': current_iter,
                  'from_addr': from_addr,  'accuracy': accuracy}
    last_block = BlockChainDB().last()
    untx_hashes = []
    cb = Block(last_block['index'] + 1, int(time.time()), untx_hashes, last_block['hash'], is_model, model_info)
    nouce = cb.pow()
    cb.make(nouce)
    # 验证现在能否插入 todo
    if valid_before_insert():
        # Save block to database.
        print('new_model:')
        print(cb.to_dict())
        BlockChainDB().insert(cb.to_dict())
        # Broadcast to other nodes
        Block.spread(cb.to_dict(), [], from_addr)
        return cb
    else:
        print("上一轮训练还未结束，无法插入！")
        pass
