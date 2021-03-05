# coding:utf-8
from account import *
from network.rpc import get_clients, BroadCast, start_server
from transaction import *
from data.database import StartDB
from block import *
import sys
from miner import *
import multiprocessing
import network.rpc
from node import *
from lib.common import cprint
import inspect
sys.path.append("..")
from FedAvg.controller import new_model_init as new_model_init
from datetime import datetime
import time
import threading

MODULES = ['account','tx','blockchain','miner','node','system']
def upper_first(string):
    return string[0].upper()+string[1:]
# 定时器任务，固定间隔更新区块链，同步节点
def time_thread(n):
    print("开始同步")
    init_node()
    t = threading.Timer(n, time_thread, (n,))
    t.start()
class Node():

    def add(self, args):
        add_node(args[0])
        rpc.BroadCast().add_node(args[0])
        cprint('Allnode',get_nodes())
    
    def run(self, args):
        if get_account() == None:
            cprint('ERROR','Please create account before start miner.')
            exit()
        sdb = StartDB()
        sdb.write(args[1])
        sdb.write(args[2])
        start_node(args[0])
        # 每5分钟更新一次节点信息
        time_thread(300)
    def list(self, args):
        for t in NodeDB().find_all():
            cprint('Node',t)





class Miner():
    def start(self, args):
        if get_account() == None:
            cprint('ERROR','Please create account before start miner.')
            exit()
        start_node(args[0])
        # 每5分钟更新一次节点信息
        time_thread(300)
        while True :
        # 挖矿暂停时间
            time.sleep(5)
            if len(args) > 1:
                default_key = args[1]
            else:
                default_key = 0
            cb = mine(default_key)
            if cb == 0:
                pass
            else:
                cprint('Miner new block',cb.to_dict())
            # timer(600)

    def new_model(self,args):
        if BlockChainDB().last()['model_info']:
            model_id = BlockChainDB().last()['model_info']['model_id'] + 1
        else:
            model_id = 1
        init_ipfs_hash, accuracy = new_model_init()
        # totoal_iter 默认迭代次数，默认为10
        cb = init_new_model_block(model_id, init_ipfs_hash, 10, accuracy)



class Account():
    def create(self, args):
        ac = new_account()
        cprint('Private Key',ac[0])
        cprint('Public Key',ac[1])
        cprint('Address',ac[2])

    def get(self, args):
        cprint('All Account',AccountDB().read())

    def current(self, args):
        cprint('Current Account', get_account())

    def getAmount(self, args):
        amount = 0
        unspents = Vout.get_unspent(args[0])
        for u in unspents:
            amount += u.get_amount()
        print("Amount:",amount)

    def list(self, args):
        for t in AccountDB().find_all():
            cprint('Account',t)
    # 用于docker测试，方便显示docker名字
    def init_name(self,args):
        adb = AccountDB()
        adb.insert({'pubkey': "8c5a38f972b307a4bd1c4aabda30685b43ad77cf", 'address': args[0]})

class Blockchain():

    def list(self, args):
        for t in BlockChainDB().find_all():
            cprint('Blockchain',str(t))

    def list_model(self, args):
        for t in BlockChainDB().find_is_model():
            cprint('Blockchain',str(t))

class Tx():

    def list(self, args):
        for t in TransactionDB().find_all():
            cprint('Transaction',t)


    def list_model(self, args):
        for t in TransactionDB().find_is_model():
            cprint('Transaction',t)

    def transfer(self, args):
        tx = Transaction.transfer(args[0], args[1], args[2])
        print(Transaction.unblock_spread(tx))
        cprint('Transaction tranfer',tx)

    def submit(self,args):
        tx = Transaction.submitModel(args[0], args[1], args[2], args[3])
        print(Transaction.unblock_spread(tx))
        cprint('new model update',tx)


class System():
    def clear(self, args):
        f1 = open('./data/account.txt', 'w')
        f1.truncate()
        # print("清空account成功")
        # f1 = open('./data/blockchain.txt', 'w')
        # f1.truncate()
        print("清空blockchain成功")
        f1 = open('./data/tx.txt', 'w')
        f1.truncate()
        print("清空tx成功")
        f1 = open('./data/untx.txt', 'w')
        f1.truncate()
        print("清空untx成功")
def usage(class_name):
    module = globals()[upper_first(class_name)]
    print('  ' + class_name + '\r')
    print('    [action]\r')
    for k,v in module.__dict__.items():
        if callable(v):
            print('      %s' % (k,))
    print('\r')

def help():
    print("Usage: python console.py [module] [action]\r")
    print('[module]\n')
    for m in MODULES:
        usage(m)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        help()
        exit()
    module = sys.argv[1]
    if module == 'help':
        help()
        exit()
    if module not in MODULES:
        cprint('Error', 'First arg shoud in %s' % (str(MODULES,)))
        exit()
    mob = globals()[upper_first(module)]()
    method = sys.argv[2]
    # try:
    getattr(mob, method)(sys.argv[3:])
    # except Exception as e:
    #     cprint('ERROR','/(ㄒoㄒ)/~~, Maybe command params get wrong, please check and try again.')