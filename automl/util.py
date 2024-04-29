import os
import datetime
import warnings
warnings.filterwarnings('ignore')
import json
import pickle
import hashlib
import gzip

import logging
logging.basicConfig(format="%(levelname)s %(asctime)s (%(filename)s %(lineno)s %(funcName)s) %(message)s",level=logging.WARNING)
logger = logging.getLogger('automl')
logger.setLevel(getattr(logging, 'DEBUG'))

def txt_reader(txt_path, extension_name=''):
    """ txt文件读取的迭代器, 可以读取文件和文件夹(此时只读取一层)

    usage: for line in txt_reader(txt_path): ...

    @params:
      - txt_path: str or list
            当类型为list时表示文件名的集合
            当类型为str表示文件路径(可以是文件名或文件夹名), 如果是文件夹, 则读取文件夹下所有后缀为extension_name的文件名
      - extension_name: 文件扩展名, 当txt_path为文件夹时, 筛选文件扩展名为extension_name的文件, 其他情况不生效
    """
    if isinstance(txt_path, list):
        txt_files = txt_path
    elif os.path.isfile(txt_path):
        txt_files = [txt_path]
    elif os.path.isdir(txt_path):
        txt_files = [os.path.join(txt_path, x) for x in next(os.walk(txt_path))[2] if x.endswith(extension_name)]
    else:
        raise Exception(f'txt path "{txt_path}" error')
    for txt_file in txt_files:
        fin = gzip.open(txt_file, 'rt') if txt_file.endswith('.gz') else open(txt_file)
        for line in fin:
            yield line
        fin.close()

def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
        
def load_json(path):
    with open(path) as f:
      return json.load(f)

def get_partition(id, num_partitions):
    # 使用MD5哈希函数计算用户ID的哈希值
    md5_hash = hashlib.md5(str(id).encode())
    # 将哈希值转换为整数
    hash_int = int(md5_hash.hexdigest(), 16)
    # 对哈希值取模得到分区编号
    partition = hash_int % num_partitions
    return partition

def get_timestamp(date, format):
    '''获取毫秒级的时间戳'''
    return datetime.datetime.strptime(date, format).timestamp()*1000

def get_datetime(date, format):
    '''时间字符串转时间类型'''
    return datetime.datetime.strptime(date, format)
