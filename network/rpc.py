# coding:utf-8
from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.client import ServerProxy
from node import get_nodes, add_node, init_node

from data.database import BlockChainDB, UnTransactionDB, TransactionDB
from lib.common import cprint
from FedAvg.controller import train as NewModelEvent
# from transaction import Transaction
import _thread
server = None

PORT = 8301

class RpcServer():

    def __init__(self,server):
        self.server = server

    def ping(self):
        return True
    
    def get_blockchain(self):
        bcdb = BlockChainDB()
        return bcdb.find_all()

    def new_block(self,block, untxs, addr):
        cprint(__name__, block)
        BlockChainDB().insert(block)
        for item in untxs:
            TransactionDB().write(item)
            UnTransactionDB().delete(item)
        # UnTransactionDB().clear()
        cprint('INFO',"Receive new block from"+addr['address'])
        # 如果是联邦块，则通知训练,上传交易
        if block['is_model'] == 1 and (block['model_info']['total_iter'] == block['model_info']['current_iter']):
            cprint('INFO', "全部迭代完成")
            cprint('INFO', "第" + str(block['model_info']['current_iter']) + "轮迭代完成")
            cprint('INFO', "共 " + str(len(block['tx'])) + ' 个节点参与')
            cprint('INFO', "准确率:" + str(block['model_info']['accuracy']))
            cprint('INFO', "Ipfs Hash:" + block['model_info']['init_ipfs_hash'])
        if block['is_model'] == 1 and (block['model_info']['total_iter'] > block['model_info']['current_iter']):
            if block['model_info']['current_iter'] == 0:
                cprint('INFO', "收到新的联邦训练任务:")
                cprint('INFO', "初始化Ipfs Hash:"+block['model_info']['init_ipfs_hash'])
                cprint('INFO', "总轮次:" + str(block['model_info']['total_iter']))
                cprint('INFO', "初始准确率:" + str(block['model_info']['accuracy']))
            else:
                cprint('INFO', "第"+str(block['model_info']['current_iter'])+"轮迭代完成")
                cprint('INFO', "共 "+str(len(block['tx']))+' 个节点参与')
                cprint('INFO', "准确率:" + str(block['model_info']['accuracy']))
                cprint('INFO', "Ipfs Hash:" + block['model_info']['init_ipfs_hash'])
                cprint('INFO', "广播，开始下一轮训练")
           # 因为xmlRPC是阻塞的，所以此时加一个线程来训练和提交交易
            try:
                _thread.start_new_thread(thread_train,(block['model_info']['model_id'], block['model_info']['current_iter']+1, block['model_info']['init_ipfs_hash'],))
            except:
                print("Error: 无法启动线程")
            # from transaction import Transaction
            # model_id, local_parameter_hash, accuracy, iter = NewModelEvent(block['model_info']['model_id'], block['model_info']['current_iter']+1, block['model_info']['init_ipfs_hash'])
            # txdic = Transaction.submitModel(model_id, local_parameter_hash, accuracy, iter)
            # print(txdic)
        return True

    def get_transactions(self):
        tdb = TransactionDB()
        return tdb.find_all()

    def new_untransaction(self,untx):
        cprint(__name__,untx)
        UnTransactionDB().insert(untx)
        cprint('INFO',"Receive new unchecked transaction from " + untx['from_addr']['address'])
        if untx['is_model'] == 1:
            cprint('INFO', "收到节点上传的模型 " + untx['from_addr']['address'])
            cprint('INFO', "准确率: " + str(untx['model_info']['accuracy']))
        return True

    # def blocked_transactions(self,txs):
    #     TransactionDB().write(txs)
    #     cprint('INFO',"Receive new blocked transactions.")
    #     return True

    def add_node(self, address):
        add_node(address)
        init_node()
        return True

    def get_all_client(self):
        clients = []
        nodes = get_nodes()
        for node in nodes:
            clients.append(node)
        return clients
class RpcClient():

    ALLOW_METHOD = ['get_transactions', 'get_blockchain', 'get_all_client', 'new_block', 'new_untransaction', 'blocked_transactions', 'ping', 'add_node']

    def __init__(self, node):
        self.node = node
        self.client = ServerProxy(node)
    
    def __getattr__(self, name):
        def noname(*args, **kw):
            if name in self.ALLOW_METHOD:
                return getattr(self.client, name)(*args, **kw)
        return noname

class BroadCast():

    def __getattr__(self, name):
        def noname(*args, **kw):
            cs = get_clients()
            rs = []
            for c in cs:
                try:
                    rs.append(getattr(c,name)(*args, **kw))
                except ConnectionRefusedError:
                    cprint('WARN', 'Contact with node %s failed when calling method %s , please check the node.' % (c.node,name))
                else:
                    cprint('INFO', 'Contact with node %s successful calling method %s .' % (c.node,name))
            return rs
        return noname

def start_server(ip, port=8301):
    server = SimpleXMLRPCServer((ip, port))
    rpc = RpcServer(server)
    server.register_instance(rpc)
    server.serve_forever()


def get_clients():
    clients = []
    nodes = get_nodes()
    for node in nodes:
       clients.append(RpcClient(node))
    return clients


def thread_train(model_id,current_iter,ipfs_hash):
    from transaction import Transaction
    model_id, local_parameter_hash, accuracy, iter = NewModelEvent(model_id,
                                                                   current_iter,
                                                                   ipfs_hash)
    txdic = Transaction.submitModel(model_id, local_parameter_hash, accuracy, iter)
    if not txdic:
        print("交易提交失败，交易不合法")
    else:
        print(txdic)