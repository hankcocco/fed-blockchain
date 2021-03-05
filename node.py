# coding:utf-8
import multiprocessing
from network import rpc
from data.database import NodeDB, TransactionDB, BlockChainDB
from lib.common import cprint


def start_node(hostport='0.0.0.0:3009'):
    init_node()
    cprint('INFO', 'Node initialize success.')
    try:
        if hostport.find('.') != -1:
            host, port = hostport.split(':')
        else:
            host = '0.0.0.0'
            port = hostport
    except Exception:
        cprint('ERROR', 'params must be {port} or {host}:{port} , ps: 3009 or 0.0.0.0:3009')
    p = multiprocessing.Process(target=rpc.start_server, args=(host, int(port)))
    p.start()
    cprint('INFO', 'Node start success. Listen at %s.' % (hostport,))


def init_node():
    """
    Download blockchain from node compare with local database and select the longest blockchain.
    """
    all_node_blockchains = rpc.BroadCast().get_blockchain()
    all_node_txs = rpc.BroadCast().get_transactions()
    all_node_nodes = rpc.BroadCast().get_all_client()
    bcdb = BlockChainDB()
    txdb = TransactionDB()
    nodedb = NodeDB()
    blockchain = bcdb.find_all()
    transactions = txdb.find_all()
    nodes = nodedb.find_all()
    # If there is a blochain downloaded longer than local database then relace local's.
    for bc in all_node_blockchains:
        if len(bc) > len(blockchain):
            blockchain = bc
            print("接收到更长区块信息")
            bcdb.clear()
            bcdb.write(bc)
    for txs in all_node_txs:
        if len(txs) > len(transactions):
            transactions = txs
            print("接收到更长交易信息")
            txdb.clear()
            txdb.write(txs)
    for node in all_node_nodes:
        nodes = list(set(nodes) | set(node))
    nodedb.clear()
    nodedb.write(nodes)


def get_nodes():
    return NodeDB().find_all()


def add_node(address):
    ndb = NodeDB()
    all_nodes = ndb.find_all()
    if address.find('http') != 0:
        address = 'http://' + address
    all_nodes.append(address)
    ndb.clear()
    ndb.write(rm_dup(all_nodes))
    return address


def check_node(address):
    pass


def rm_dup(nodes):
    return sorted(set(nodes))


if __name__ == '__main__':
    start_node(3009)
