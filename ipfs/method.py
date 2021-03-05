"""
     pip install ipfshttpclient==0.7.0a1
"""
import ipfshttpclient as ipfs
import os
import shutil
try:
    # 针对宿主机
    # api = ipfs.connect('/ip4/127.0.0.1/tcp/5001')
    # 针对docker容器， 其中192.*需要查看, 需要重新配置ipfs-api
    api = ipfs.connect('/ip4/172.31.233.230/tcp/5001')
except Exception as e:
    print(e)
    print("没有检测到ipfs deamon")
    pass


def ipfs_add(absolutely_path):
    res = api.add(absolutely_path)
    return res['Hash']


def ipfs_get(src_hash, target_path):
    api.get(src_hash)
    work_dir = os.getcwd()
    src = os.path.join(work_dir, src_hash)
    shutil.move(src, target_path)
    return target_path

