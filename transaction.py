"""
交易结构文件
包含交易生成、验证的一些固有方法
目前还不能接受来自其他节点的未打包交易
"""
# coding:utf-8
import time
import hashlib
from lib.model import Model
from data.database import TransactionDB, UnTransactionDB, AccountDB
from network.rpc import BroadCast
class Vin(Model):
    def __init__(self, utxo_hash, amount):
        self.hash = utxo_hash
        self.amount = amount
        # self.unLockSig = unLockSig
    def get_amount(self):
        return self.amount
class Vout(Model):
    def __init__(self, receiver, amount):
        self.receiver = receiver
        self.amount = amount
        self.hash = hashlib.sha256((str(time.time()) + str(self.receiver) + str(self.amount)).encode('utf-8')).hexdigest()
        # self.lockSig = lockSig
    
    @classmethod
    def get_unspent(cls, addr):
        """
        Exclude all consumed VOUT, get unconsumed VOUT
        
        """
        unspent = []
        all_tx = TransactionDB().find_all()
        spend_vin = []
        [spend_vin.extend(item['vin']) for item in all_tx]
        has_spend_hash = [vin['hash'] for vin in spend_vin]
        for item in all_tx:
            # Vout receiver is addr and the vout hasn't spent yet.
            # 地址匹配且未花费
            for vout in item['vout']:
                if vout['receiver'] == addr and vout['hash'] not in has_spend_hash:
                    unspent.append(vout)
        return [Vin(tx['hash'], tx['amount']) for tx in unspent]

class Transaction():
    # 针对转账交易初始化
    """
    @param is_model = 0表示这是正常转账交易 =1表示是模型更新
    @param model_info：
    [
        'model_id'          学习的联邦id
        'model_ipfs_hash'   上传的模型ipfs地址
        'accuracy'          模型准确率
        'iter_num'          第几轮联邦迭代
    ]
    """
    def __init__(self, vin, vout, from_addr, is_model=0, model_info=None):
        if model_info is None:
            model_info = {}
        self.timestamp = int(time.time())
        self.vin = vin
        self.vout = vout
        self.hash = self.gen_hash()
        # 增加一个交易发起人字段，目前没有严格校验
        self.from_addr = from_addr
        self.is_model = is_model
        self.model_info = model_info

    def gen_hash(self):
        return hashlib.sha256((str(self.timestamp) + str(self.vin) + str(self.vout)).encode('utf-8')).hexdigest()


    @classmethod
    # 联邦学习客户端提交模型
    # todo 验证上传的值
    def submitModel(cls, model_id, model_ipfs_hash, accuracy, iter_num):
        # 上传节点，目前默认为第一个地址
        from_addr = AccountDB().find_some(0)
        # todo 验证模型的轮次和id是否正确
        if valid_model(model_id, iter_num) == 1:
            vin = []
            vout = []
            model_info = {'model_id': model_id, 'model_ipfs_hash': model_ipfs_hash, 'accuracy': accuracy, 'iter_num': iter_num}
            tx = cls(vin, vout, from_addr, 1, model_info)
            tx_dict = tx.to_dict()
            UnTransactionDB().insert(tx_dict)
            Transaction.unblock_spread(tx_dict)
            return tx_dict
        else:
            return False

    @classmethod
    def transfer(cls, from_addr, to_addr, amount):
        if not isinstance(amount,int):
            amount = int(amount)
        # unspents是所有未消费的vout
        unspents = Vout.get_unspent(from_addr)
        # ready_utxo ,是准备消费的vout表，  change是消费后的找零
        ready_utxo, change = select_outputs_greedy(unspents, amount)
        print('ready_utxo', ready_utxo[0].to_dict())

        vin = ready_utxo
        vout = []
        # 把账户钱全部花出去，比如转账18，账户原有20.那么18到to_addr，2再自己转账自己
        vout.append(Vout(to_addr, amount))
        vout.append(Vout(from_addr, change))
        # 这边vin,vout构成交易后，代表vin被花费掉，vout是收入
        tx = cls(vin, vout, from_addr)
        tx_dict = tx.to_dict()
        UnTransactionDB().insert(tx_dict)
        Transaction.unblock_spread(tx_dict)
        return tx_dict

    @staticmethod
    def unblock_spread(untx):
        BroadCast().new_untransaction(untx)
    # @staticmethod
    # def blocked_spread(txs):
    #     BroadCast().blocked_transactions(txs)

    def to_dict(self):
        dt = self.__dict__
        if not isinstance(self.vin, list):
            self.vin = [self.vin]
        if not isinstance(self.vout, list):
            self.vout = [self.vout]
        dt['from_addr'] = self.from_addr
        dt['vin'] = [i.__dict__ for i in self.vin]
        dt['vout'] = [i.__dict__ for i in self.vout]
        dt['is_model'] = self.is_model
        dt['model_info'] = self.model_info
        return dt
def valid_model(from_addr,model_id):
    # todo 验证上传模型迭代轮次、模型id是否正确
    return 1
# 找到符合转账金额的未消费vout
def select_outputs_greedy(unspent, min_value): 
    if not unspent: return None 
    # 分割成两个列表。
    lessers = [utxo for utxo in unspent if utxo.amount < min_value] 
    greaters = [utxo for utxo in unspent if utxo.amount >= min_value] 
    key_func = lambda utxo: utxo.amount
    greaters.sort(key=key_func)
    if greaters: 
        # 非空。寻找最小的greater。
        min_greater = greaters[0]
        change = min_greater.amount - min_value 
        return [min_greater], change
    # 没有找到greaters。重新尝试若干更小的。
    # 从大到小排序。我们需要尽可能地使用最小的输入量。
    lessers.sort(key=key_func, reverse=True)
    result = []
    accum = 0
    for utxo in lessers: 
        result.append(utxo)
        accum += utxo.amount
        if accum >= min_value: 
            change = accum - min_value
            return result, change 
    # 没有找到。
    return None, 0