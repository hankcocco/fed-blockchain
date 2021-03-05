"""
各数据结构存储逻辑
"""
# coding:utf-8
import json
import os

BASEDBPATH = 'data'
BLOCKFILE = 'blockchain.txt'
TXFILE = 'tx.txt'
UNTXFILE = 'untx.txt'
ACCOUNTFILE = 'account.txt'
NODEFILE = 'node.txt'
STARTFILE = 'startend.txt'

class BaseDB():
    filepath = ''

    def __init__(self):
        self.set_path()
        self.filepath = '/'.join((BASEDBPATH, self.filepath))

    def set_path(self):
        pass

    def find_all(self):
        return self.read()

    def insert(self, item):
        self.write(item)

    def read(self):
        raw = ''
        if not os.path.exists(self.filepath):
            return []
        with open(self.filepath, 'r+') as f:
            raw = f.readline()
        if len(raw) > 0:
            data = json.loads(raw)
        else:
            data = []
        return data

    def delete(self, item):
        re_write = []
        all = self.find_all()
        for data in all:
            if data != item:
                re_write.append(data)
        self.clear()
        for data in re_write:
            self.write(data)

    def write(self, item):
        data = self.read()
        if isinstance(item, list):
            data = data + item
        else:
            data.append(item)
        with open(self.filepath, 'w+') as f:
            f.write(json.dumps(data))
        return True

    def clear(self):
        with open(self.filepath, 'w+') as f:
            f.write('')

    def hash_insert(self, item):
        exists = False
        for i in self.find_all():
            if item['hash'] == i['hash']:
                exists = True
                break
        if not exists:
            self.write(item)


class NodeDB(BaseDB):

    def set_path(self):
        self.filepath = NODEFILE

class StartDB(BaseDB):
    def set_path(self):
        self.filepath = STARTFILE
    def get(self):
        ac = self.read()
        start = int(ac[0])
        end = int(ac[1])
        return start, end
class AccountDB(BaseDB):
    def set_path(self):
        self.filepath = ACCOUNTFILE

    def find_one(self):
        ac = self.read()
        return ac[0]

    def find_some(self, key):
        ac = self.read()
        return ac[key]


class BlockChainDB(BaseDB):

    def set_path(self):
        self.filepath = BLOCKFILE

    def last(self):
        bc = self.read()
        if len(bc) > 0:
            return bc[-1]
        else:
            return []

    def find(self, hash):
        one = {}
        for item in self.find_all():
            if item['hash'] == hash:
                one = item
                break
        return one

    def insert(self, item):
        self.hash_insert(item)

    def find_is_model(self):
        result = []
        for item in self.find_all():
            if item['is_model'] == 1:
                result.append(item)
        return result

    def find_not_model(self):
        result = []
        for item in self.find_all():
            if item['is_model'] != 1:
                result.append(item)
        return result


class TransactionDB(BaseDB):
    """
    Transactions that save with blockchain.
    """

    def set_path(self):
        self.filepath = TXFILE

    def find(self, hash):
        one = {}
        for item in self.find_all():
            if item['hash'] == hash:
                one = item
                break
        return one

    def insert(self, txs):
        if not isinstance(txs, list):
            txs = [txs]
        for tx in txs:
            self.hash_insert(tx)

    def find_is_model(self):
        result = []
        for item in self.find_all():
            if item['is_model'] == 1:
                result.append(item)
        return result

    def find_not_model(self):
        result = []
        for item in self.find_all():
            if item['is_model'] == 0:
                result.append(item)
        return result


class UnTransactionDB(TransactionDB):
    """
    Transactions that doesn't store in blockchain.
    """

    def set_path(self):
        self.filepath = UNTXFILE

    def all_hashes(self):
        hashes = []
        for item in self.find_all():
            hashes.append(item['hash'])
        return hashes

    def is_model_hashes(self):
        hashes = []
        for item in self.find_is_model():
            hashes.append(item['hash'])
        return hashes

    def not_model_hashes(self):
        hashes = []
        for item in self.find_not_model():
            hashes.append(item['hash'])
        return hashes


    def find_is_model(self):
        result = []
        for item in self.find_all():
            if item['is_model'] == 1:
                result.append(item)
        return result


    def find_not_model(self):
        result = []
        for item in self.find_all():
            if item['is_model'] == 0:
                result.append(item)
        return result