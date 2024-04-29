import argparse
import yaml
import sys
from automl.feature_extract.feature_config import *
from automl.feature_extract.feature_extractor import *


import importlib

def load_version(version):
    try:
        module = importlib.import_module(f'my_module_v{version}')
    except ImportError:
        raise ImportError(f"Version {version} of the module is not found.")
    return module

def call_function_from_path(func):
    """通过指定函数所在的文件路径来获取并执行该文件中的函数"""
    if not func: return None
    module_path, function_name = func['py_file_path'], func['function_name']
    spec = importlib.util.spec_from_file_location("custom_module", module_path)
    custom_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_module)
    function = getattr(custom_module, function_name)
    return function

def main():
    # 1. 获取命令行配置文件参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./features.yaml')
    args = parser.parse_args()
    with open(args.conf, 'r') as fin:
        yaml_reader = yaml.load(fin, Loader=yaml.FullLoader)
    # 2. 解析配置文件，获取特征衍生参数
    featureconfig = FeatureConfigHelper(yaml_reader)
    params = featureconfig.get_feature_extract_params()
    num_partitions = yaml_reader['num_partitions']
    custom_func = call_function_from_path(yaml_reader.get('custom_function_features'))
    # 3. 特征抽取
    FeatureExtractorV2(**params).repartition(num_partitions).sort_by_partition().map_partition(custom_func)
    


if __name__ == '__main__':
    main()