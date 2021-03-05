"""
区块结构文件
包含共识、存储、验证
todo 节点对新到来区块进行验证
"""
# coding:utf-8
import hashlib
from lib.model import Model
from network.rpc import BroadCast

class Block(Model):

    def __init__(self, index, timestamp, tx, previous_hash, is_model=0, model_info=None):
        if model_info is None:
            model_info = {}
        self.index = index
        self.timestamp = timestamp
        self.tx = tx
        self.previous_block = previous_hash
        self.is_model = is_model
        self.model_info = model_info
    def header_hash(self):
        """
        Refer to bitcoin block header hash
        计算区块哈希头
        """

        return hashlib.sha256((str(self.index) + str(self.timestamp) + str(self.tx) + str(self.previous_block)).encode('utf-8')).hexdigest()

    def pow(self):
        """
        Proof of work. Add nouce to block.
        不断增加nouce的值，使得header_hash+nouce后的hash，前n位是0
        """        
        nouce = 0
        while self.valid(nouce) is False:
            nouce += 1
        self.nouce = nouce
        return nouce

    def make(self, nouce):
        """
        Block hash generate. Add hash to block.
        """
        self.hash = self.ghash(nouce)
    
    def ghash(self, nouce):
        """
        Block hash generate.
        """        
        header_hash = self.header_hash()
        token = f'{header_hash}{nouce}'.encode('utf-8')
        return hashlib.sha256(token).hexdigest()

    def valid(self, nouce):
        """
        Validates the Proof
        todo valid more info
        """
        return self.ghash(nouce)[:4] == "0000"

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, bdict):
        b = cls(bdict['index'], bdict['timestamp'], bdict['tx'], bdict['previous_block'], bdict['is_model'], bdict['model_info'])
        b.hash = bdict['hash']
        b.nouce = bdict['nouce']
        return b

    @staticmethod
    def spread(block, untxs,from_addr):
        BroadCast().new_block(block, untxs, from_addr)