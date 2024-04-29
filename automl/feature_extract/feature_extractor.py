import os
import time
import itertools
import pandas as pd

from automl.util import *
from automl.feature_extract.feature_util import *
from automl.feature_extract.CONST import *
from automl.feature_extract.feature_instance import *

class FeatureExtractor:
    '''特征提取类，实现样本表、特征数据表之间按照配置提取相关的特征
    
    参数：
        - main_table_file: 样本表文件路径
        - main_table_file_extension: 文件扩展名
        - main_table_sep: 文件分隔符
        - main_partition_col: 分区列(也指key)
        - main_sort_col: 时间列
        - mode: 字符串，单表特征衍生还是双表特征衍生
        - feat_table_file: 特征数据文件路径
        - feat_table_file_extension: 文件扩展名
        - feat_table_sep: 文件分隔符
        - feat_partition_col: 分区列(也指key)
        - feat_sort_col: 时间列
        - output_file: 输出特征文件名称
        - calc_config_file: json文件, 需要配置的计算列、计算函数、条件筛选
    '''
    def __init__(
        self,
        main_table_file: str=None, 
        main_schema_file: str=None,
        main_table_file_extension: str='.csv',
        main_table_sep: str=',', 
        main_partition_col: str=None, 
        main_sort_col: str=None,
        
        mode: str='two',

        feat_table_file: str=None, 
        feat_schema_file: str=None,
        feat_table_file_extension: str='.csv',
        feat_table_sep: str=',', 
        feat_partition_col: str=None, 
        feat_sort_col: str=None,
        
        output_file: str=None,
        calc_config_file: str = None,
        tmp_dir: str = './tmp_feature',
        log: bool = False
    ):
        self.main_table_file = main_table_file
        self.main_schema_file = main_schema_file
        self.main_schema = parse_schema(self.main_schema_file)
        # 检查数据类型
        check_data_type(main_schema_file, DATA_TYPE)
        self.main_table_file_extension = main_table_file_extension
        self.main_table_sep = main_table_sep
        self.main_partition_col = main_partition_col
        self.main_sort_col = main_sort_col

        if mode not in ['one', 'two']:
            raise Exception(f"unsupported mode {mode}, expected 'one' or 'two'")
        self.mode = mode
        if mode == 'two':
            assert feat_table_file is not None, f'parameter feat_table_file should not be None'
            assert feat_table_file_extension is not None, f'parameter feat_table_file_extension should not be None'
            assert feat_table_sep is not None, f'parameter feat_table_sep should not be None'
            assert feat_partition_col is not None, f'parameter feat_partition_col should not be None'
            assert feat_sort_col is not None, f'parameter feat_sort_col should not be None'
            assert feat_schema_file is not None, f'parameter feat_schema_file should not be None'
        
        self.feat_table_file = feat_table_file
        self.feat_schema_file = feat_schema_file
        if feat_schema_file:
            self.feat_schema = parse_schema(self.feat_schema_file)
            # 检查数据类型
            check_data_type(feat_schema_file, DATA_TYPE)
        self.feat_table_file_extension = feat_table_file_extension
        self.feat_table_sep = feat_table_sep
        self.feat_partition_col = feat_partition_col
        self.feat_sort_col = feat_sort_col
        
        self.output_file = output_file
        self.calc_config_file = calc_config_file
        
        if calc_config_file:
            self.configs = load_json(calc_config_file)
            parse_configs = parse_calc_config(self.configs)
            self.trans_configs = trans_config(parse_configs)
        else:
            self.trans_configs = None

        self.tmp_dir = tmp_dir
        self.log = log
        # self.tmp_file_prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tmp_file_prefix = "feature_extract"
        self.start_time = time.time()

    def repartition(self, num_partitions=1):
        '''将数据按照指定的partition列进行分区,使得同一个partition的数据在同一个分区中(如果num_partitions=1,说明数据量较小直接加载内存处理)

        参数：
            - num_partitions: 分区文件个数
        '''
        self.num_partitions = num_partitions
        if self.mode == 'one':
            return self.repartition_with_one_table()
        elif self.mode == 'two':
            return self.repartition_with_two_table()
        
    def repartition_with_one_table(self):
        num_partitions = self.num_partitions
        main_partition_col, main_sort_col = self.main_partition_col, self.main_sort_col
        main_table_sep = self.main_table_sep

        output_dir = self.tmp_dir

        # 创建输出文件夹
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        # 创建输出文件
        output_files = {}
        for i in range(num_partitions):
            output_file_path = os.path.join(output_dir, f'{self.tmp_file_prefix}_num_partition_{i+1}.csv')
            output_files[i] = open(output_file_path, 'w')
        
        logger.info('split data file start')
        main_cols = split_file_by_partition_col(self.main_table_file, main_table_sep, main_partition_col, 
                                                main_sort_col, '', num_partitions, output_files)
        logger.info(f'split data file end, file num = {num_partitions}')

        # 关闭所有输出文件
        for i in range(num_partitions):
            output_files[i].close()
        
        self.main_cols = main_cols.split(main_table_sep)
        return self
    
    def repartition_with_two_table(self):
        num_partitions = self.num_partitions
        main_partition_col, main_sort_col = self.main_partition_col, self.main_sort_col
        main_table_sep = self.main_table_sep
        feat_partition_col, feat_sort_col = self.feat_partition_col, self.feat_sort_col
        feat_table_sep = self.feat_table_sep

        output_dir = self.tmp_dir

        # 创建输出文件夹
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        # 创建输出文件
        output_files = {}
        for i in range(num_partitions):
            output_file_path = os.path.join(output_dir, f'{self.tmp_file_prefix}_num_partition_{i+1}.csv')
            output_files[i] = open(output_file_path, 'w')
        
        logger.info('split data file start')
        main_cols = split_file_by_partition_col(self.main_table_file, main_table_sep, main_partition_col, 
                                                main_sort_col, 1, num_partitions, output_files)
        feat_cols = split_file_by_partition_col(self.feat_table_file, feat_table_sep, feat_partition_col, 
                                                feat_sort_col, 0, num_partitions, output_files)
        logger.info(f'split data file end, file num = {num_partitions}')

        # 关闭所有输出文件
        for i in range(num_partitions):
            output_files[i].close()
        
        self.main_cols, self.feat_cols = main_cols.split(main_table_sep), feat_cols.split(feat_table_sep)
        return self
        
    
    def sort_by_partition(self):
        '''
        参数：
            - num_partitions: 分片文件数
        '''
        if self.mode == 'one':
            return self.sort_by_partition_with_one_table()
        elif self.mode == 'two':
            return self.sort_by_partition_with_two_table()
    
    def sort_by_partition_with_one_table(self):
        num_partitions = self.num_partitions
        main_sort_col, main_schema = self.main_sort_col, self.main_schema

        for i in range(num_partitions):
            logger.info(f'sort {i+1} partition data start')
            data = []
            with open(self.tmp_dir+f'/{self.tmp_file_prefix}_num_partition_{i+1}.csv', 'r') as fin:
                for line in fin:
                    sort_val, vals = line.strip().split('|')
                    sort_val_lst, vals_lst = sort_val.split(','), vals.split(',')
                    timestamp = get_timestamp(sort_val_lst[1], main_schema[main_sort_col][1])
                    data_lst = transformer_data(main_schema, vals_lst)
                    data.append((sort_val_lst[0], int(timestamp), data_lst))
            data.sort(key=lambda x: (x[0], x[1]))
            save_obj(data, self.tmp_dir+f'/{self.tmp_file_prefix}_num_partition_{i+1}_sorted')
            logger.info(f'sort {i+1} partition data end, data size = {len(data)}')
        return self

    def sort_by_partition_with_two_table(self):
        num_partitions = self.num_partitions
        main_sort_col, feat_sort_col = self.main_sort_col, self.feat_sort_col
        main_schema, feat_schema = self.main_schema, self.feat_schema

        for i in range(num_partitions):
            logger.info(f'sort partition {i+1} data start')
            data = []
            with open(self.tmp_dir+f'/{self.tmp_file_prefix}_num_partition_{i+1}.csv', 'r') as fin:
                for line in fin:
                    sort_val, vals = line.strip().split('|')
                    sort_val_lst, vals_lst = sort_val.split(','), vals.split(',')
                    if int(sort_val_lst[2]) == 0: # 特征表
                        timestamp = get_timestamp(sort_val_lst[1], feat_schema[feat_sort_col][1])
                        data_lst = transformer_data(feat_schema, vals_lst)
                    else: # 样本表
                        timestamp = get_timestamp(sort_val_lst[1], main_schema[main_sort_col][1])
                        data_lst = transformer_data(main_schema, vals_lst)

                    data.append((sort_val_lst[0], int(timestamp), int(sort_val_lst[2]), data_lst))
            data.sort(key=lambda x: (x[0], x[1], x[2]))
            save_obj(data, self.tmp_dir+f'/{self.tmp_file_prefix}_num_partition_{i+1}_sorted')
            logger.info(f'sort partition {i+1} data end, data size = {len(data)}')
        return self


    def map_partition(self, func=None):
        if self.mode == 'one':
            return self.map_partition_with_one_table(func)
        elif self.mode == 'two':
            return self.map_partition_with_two_table(func)
    
    def map_partition_with_one_table(self, func):
        output_file, configs = self.output_file, self.trans_configs
        main_cols, main_schema = self.main_cols, self.main_schema

        log, log_data, write_flag = self.log, None, 'w'
        log_file = self.tmp_dir+f'/feat_extract_log.csv'
        if log:
            if os.path.exists(log_file):
                with open(log_file, 'r') as fin:
                    log_data = [f.strip() for f in fin]
            write_flag = 'a'
        
        down_schmea, cnt = True, 0
        with open(output_file, write_flag) as feat_out, open(log_file, write_flag) as log_out:
            for i in range(self.num_partitions):
                logger.info(f'extract feature partition {i+1} data start')
                tmp_file = self.tmp_dir+f'/{self.tmp_file_prefix}_num_partition_{i+1}_sorted'
                data = load_obj(tmp_file)
                for _, groups in itertools.groupby(data, key=lambda x: x[0]):
                    window_data = []
                    for _, sort, vals in groups:
                        if log_data and '_'.join(vals) in log_data: # 如果已经计算了该样本，直接跳过
                            down_schmea = False
                            continue

                        start = time.time()
                        window_data.append([sort]+transformer_data(main_schema, vals, main_cols))
                        featurebuffer = FeatureBuffer(main_cols, vals)
                        w_df = pd.DataFrame(window_data, columns=['sort']+main_cols)
                        ins_dict = dict(zip(main_cols, transformer_data(main_schema, vals, main_cols)))

                        if func: # 自定义特征计算
                            func(featurebuffer, w_df, ins_dict)
                        if configs: # 自动化配置特征计算
                            get_feature(featurebuffer, sort, w_df, ins_dict, configs)
                        if down_schmea:
                            feat_out.write(','.join(featurebuffer.get_feature_names()))
                            feat_out.write('\n')
                            down_schmea = False
                        feat_out.write(','.join(str(x) for x in featurebuffer.get_feature_vals()))
                        feat_out.write('\n')
                        feat_out.flush()
                        # 存储已处理的样本
                        if log:
                            log_out.write('_'.join(vals))
                            log_out.write('\n')
                            log_out.flush()

                        end = time.time()
                        cnt += 1
                        logger.info(f'{cnt}th sample feature extract finished,  cost {round(end-start, 4)}s')

                logger.info(f'extract feature partition {i+1} data end')
        end_time = time.time()
        logger.info(f'task finished, cost {round(end_time-self.start_time, 4)}s')
        
        if log: # 删除日志文件
            os.remove(log_file)

    def map_partition_with_two_table(self, func):
        output_file, configs = self.output_file, self.trans_configs
        main_cols, feat_cols = self.main_cols, self.feat_cols
        # main_schema, feat_schema = self.main_schema, self.feat_schema

        log, log_data, write_flag = self.log, None, 'w'
        log_file = self.tmp_dir+f'/feat_extract_log.csv'
        if log:
            if os.path.exists(log_file):
                with open(log_file, 'r') as fin:
                    log_data = [f.strip() for f in fin]
            write_flag = 'a'
        
        down_schmea, cnt = True, 0
        with open(output_file, write_flag) as feat_out, open(log_file, write_flag) as log_out:
            for i in range(self.num_partitions):
                logger.info(f'extract feature partition {i+1} data start')
                tmp_file = self.tmp_dir+f'/{self.tmp_file_prefix}_num_partition_{i+1}_sorted'
                data = load_obj(tmp_file)
                for _, groups in itertools.groupby(data, key=lambda x: x[0]):
                    window_data = []
                    for _, sort, flag, vals in groups:
                        if flag == 0: # 特征
                            # window_data.append([sort]+transformer_data(feat_schema, vals, feat_cols))
                            window_data.append([sort]+vals)
                        else: # 样本
                            if log_data and '_'.join(vals) in log_data: # 如果已经计算了该样本，直接跳过
                                down_schmea = False
                                continue
                            
                            start = time.time()
                            featurebuffer = FeatureBuffer(main_cols, vals)
                            w_df = pd.DataFrame(window_data, columns=[0]+feat_cols)
                            # ins_dict = dict(zip(main_cols, transformer_data(main_schema, vals, main_cols)))
                            ins_dict = dict(zip(main_cols, vals))

                            if func: # 自定义特征计算
                                func(featurebuffer, pd.DataFrame(window_data, columns=[0]+feat_cols), ins_dict)
                            if configs: # 自动化配置特征计算
                                get_feature(featurebuffer, sort, w_df, ins_dict, configs)
                            if down_schmea:
                                feat_out.write(','.join(featurebuffer.get_feature_names()))
                                feat_out.write('\n')
                                down_schmea = False
                            feat_out.write(','.join(str(x) for x in featurebuffer.get_feature_vals()))
                            feat_out.write('\n')
                            feat_out.flush()
                            if log: # 存储已处理的样本
                                log_out.write('_'.join(vals))
                                log_out.write('\n')
                                log_out.flush()
                            
                            end = time.time()
                            cnt += 1
                            logger.info(f'{cnt}th sample feature extract finished,  cost {round(end-start, 4)}s')
                logger.info(f'extract feature partition {i+1} data end')
        
        end_time = time.time()
        logger.info(f'task finished, cost {round(end_time-self.start_time, 4)}s')

        if log: # 删除日志文件
            os.remove(log_file)


########################################################################################################### 

class FeatureExtractorV2:
    '''特征提取类，实现样本表、特征数据表之间按照配置提取相关的特征
    
    参数：
        - main_table_file: 样本表文件路径
        - main_table_file_extension: 文件扩展名
        - main_table_sep: 文件分隔符
        - main_partition_col: 分区列(也指key)
        - main_sort_col: 时间列
        - mode: 字符串，单表特征衍生还是双表特征衍生
        - feat_table_file: 特征数据文件路径
        - feat_table_file_extension: 文件扩展名
        - feat_table_sep: 文件分隔符
        - feat_partition_col: 分区列(也指key)
        - feat_sort_col: 时间列
        - output_file: 输出特征文件名称
        - calc_config_file: json文件, 需要配置的计算列、计算函数、条件筛选
    '''
    def __init__(
        self,
        main_table_file: str=None, 
        main_schema: dict=None,
        main_table_sep: str=',', 
        main_partition_col: str=None, 
        main_sort_col: str=None,
        main_header: bool=True,
        main_table_file_extension_name: str='.csv',
        mode: int=2,
        feat_table_file: str=None, 
        feat_schema: dict=None,
        feat_table_sep: str=',', 
        feat_partition_col: str=None, 
        feat_sort_col: str=None,
        feat_header: bool=True,
        feat_table_file_extension_name: str='.csv',
        output_file: str=None,
        output_sep: str=',',
        features: dict = None,
        tmp_dir: str = './tmp_feature',
        log: bool = False
    ):
        self.main_table_file = main_table_file
        self.main_schema = main_schema
        self.main_table_sep = main_table_sep
        self.main_partition_col = main_partition_col
        self.main_sort_col = main_sort_col
        self.main_header = main_header
        self.main_table_file_extension_name = main_table_file_extension_name
        self.main_cols = list(main_schema.keys()) # 样本表列名称
        self.mode = mode
        self.feat_table_file = feat_table_file
        self.feat_schema = feat_schema
        self.feat_table_sep = feat_table_sep
        self.feat_partition_col = feat_partition_col
        self.feat_sort_col = feat_sort_col
        self.feat_header = feat_header
        self.feat_table_file_extension_name = feat_table_file_extension_name
        self.feat_cols = list(feat_schema.keys()) # 特征表列名称
        self.output_file = output_file
        self.output_sep = output_sep
        self.features = features
        self.tmp_dir = tmp_dir
        self.log = log
        # self.tmp_file_prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tmp_file_prefix = "feature_extract"
        self.start_time = time.time()

    def repartition(self, num_partitions=1):
        '''将数据按照指定的partition列进行分区,使得同一个partition的数据在同一个分区中(如果num_partitions=1,说明数据量较小直接加载内存处理)

        参数：
            - num_partitions: 分区文件个数
        '''
        self.num_partitions = num_partitions
        if self.mode == 1:
            return self.repartition_with_one_table()
        elif self.mode == 2:
            return self.repartition_with_two_table()
        
    def repartition_with_one_table(self):
        num_partitions = self.num_partitions
        main_partition_col, main_sort_col = self.main_partition_col, self.main_sort_col
        main_table_sep, main_header = self.main_table_sep, self.main_header
        main_extension_name = self.main_table_file_extension_name

        output_dir = self.tmp_dir

        # 创建输出文件夹
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        # 创建输出文件
        output_files = {}
        for i in range(num_partitions):
            output_file_path = os.path.join(output_dir, f'{self.tmp_file_prefix}_num_partition_{i+1}.csv')
            output_files[i] = open(output_file_path, 'w')
        
        logger.info('split data file start')
        split_file_by_partition_col(self.main_table_file, main_extension_name, main_table_sep, main_partition_col, 
                                    main_sort_col, main_header, '', num_partitions, output_files)
        logger.info(f'split data file end, file num = {num_partitions}')

        # 关闭所有输出文件
        for i in range(num_partitions):
            output_files[i].close()
        
        # self.main_cols = main_cols.split(main_table_sep)
        return self
    
    def repartition_with_two_table(self):
        num_partitions = self.num_partitions
        main_partition_col, main_sort_col = self.main_partition_col, self.main_sort_col
        main_table_sep, main_header = self.main_table_sep, self.main_header
        main_extension_name, main_cols = self.main_table_file_extension_name, self.main_cols
        feat_partition_col, feat_sort_col = self.feat_partition_col, self.feat_sort_col
        feat_table_sep, feat_header = self.feat_table_sep, self.feat_header
        feat_extension_name, feat_cols = self.feat_table_file_extension_name, self.feat_cols

        output_dir = self.tmp_dir

        # 创建输出文件夹
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        # 创建输出文件
        output_files = {}
        for i in range(num_partitions):
            output_file_path = os.path.join(output_dir, f'{self.tmp_file_prefix}_num_partition_{i+1}.csv')
            output_files[i] = open(output_file_path, 'w')
        
        logger.info('split data file start')
        split_file_by_partition_col(self.main_table_file, main_extension_name, main_cols, main_table_sep, main_partition_col, 
                                    main_sort_col, main_header, 1, num_partitions, output_files)
        split_file_by_partition_col(self.feat_table_file, feat_extension_name, feat_cols, feat_table_sep, feat_partition_col, 
                                    feat_sort_col, feat_header, 0, num_partitions, output_files)
        logger.info(f'split data file end, file num = {num_partitions}')

        # 关闭所有输出文件
        for i in range(num_partitions):
            output_files[i].close()
        
        # self.main_cols, self.feat_cols = main_cols.split(main_table_sep), feat_cols.split(feat_table_sep)
        return self
        
    
    def sort_by_partition(self):
        '''
        参数：
            - num_partitions: 分片文件数
        '''
        if self.mode == 1:
            return self.sort_by_partition_with_one_table()
        elif self.mode == 2:
            return self.sort_by_partition_with_two_table()
    
    def sort_by_partition_with_one_table(self):
        num_partitions = self.num_partitions
        main_sort_col, main_schema = self.main_sort_col, self.main_schema

        for i in range(num_partitions):
            logger.info(f'sort {i+1} partition data start')
            data = []
            with open(self.tmp_dir+f'/{self.tmp_file_prefix}_num_partition_{i+1}.csv', 'r') as fin:
                for line in fin:
                    sort_val, vals = line.strip().split('|')
                    sort_val_lst, vals_lst = sort_val.split(','), vals.split(',')
                    timestamp = get_timestamp(sort_val_lst[1], main_schema[main_sort_col][1])
                    data_lst = transformer_data(main_schema, vals_lst)
                    data.append((sort_val_lst[0], int(timestamp), data_lst))
            data.sort(key=lambda x: (x[0], x[1]))
            save_obj(data, self.tmp_dir+f'/{self.tmp_file_prefix}_num_partition_{i+1}_sorted')
            logger.info(f'sort {i+1} partition data end, data size = {len(data)}')
        return self

    def sort_by_partition_with_two_table(self):
        num_partitions = self.num_partitions
        main_sort_col, feat_sort_col = self.main_sort_col, self.feat_sort_col
        main_schema, feat_schema = self.main_schema, self.feat_schema

        for i in range(num_partitions):
            logger.info(f'sort partition {i+1} data start')
            data = []
            with open(self.tmp_dir+f'/{self.tmp_file_prefix}_num_partition_{i+1}.csv', 'r') as fin:
                for line in fin:
                    sort_val, vals = line.strip().split('|')
                    sort_val_lst, vals_lst = sort_val.split(','), vals.split(',')
                    if int(sort_val_lst[2]) == 0: # 特征表
                        timestamp = get_timestamp(sort_val_lst[1], feat_schema[feat_sort_col][1])
                        data_lst = transformer_data(feat_schema, vals_lst)
                    else: # 样本表
                        timestamp = get_timestamp(sort_val_lst[1], main_schema[main_sort_col][1])
                        data_lst = transformer_data(main_schema, vals_lst)

                    data.append((sort_val_lst[0], int(timestamp), int(sort_val_lst[2]), data_lst))
            data.sort(key=lambda x: (x[0], x[1], x[2]))
            save_obj(data, self.tmp_dir+f'/{self.tmp_file_prefix}_num_partition_{i+1}_sorted')
            logger.info(f'sort partition {i+1} data end, data size = {len(data)}')
        return self


    def map_partition(self, func=None):
        if self.mode == 1:
            return self.map_partition_with_one_table(func)
        elif self.mode == 2:
            return self.map_partition_with_two_table(func)
    
    def map_partition_with_one_table(self, func):
        output_file, features = self.output_file, self.features
        main_cols, main_schema = self.main_cols, self.main_schema
        output_sep = self.output_sep

        log, log_data, write_flag = self.log, None, 'w'
        log_file = self.tmp_dir+f'/feat_extract_log.csv'
        if log:
            if os.path.exists(log_file):
                with open(log_file, 'r') as fin:
                    log_data = [f.strip() for f in fin]
            write_flag = 'a'
        
        down_schmea, cnt = True, 0
        with open(output_file, write_flag) as feat_out, open(log_file, write_flag) as log_out:
            for i in range(self.num_partitions):
                logger.info(f'extract feature partition {i+1} data start')
                tmp_file = self.tmp_dir+f'/{self.tmp_file_prefix}_num_partition_{i+1}_sorted'
                data = load_obj(tmp_file)
                for _, groups in itertools.groupby(data, key=lambda x: x[0]):
                    window_data = []
                    for _, sort, vals in groups:
                        if log_data and '_'.join(vals) in log_data: # 如果已经计算了该样本，直接跳过
                            down_schmea = False
                            continue

                        start = time.time()
                        window_data.append([sort]+transformer_data(main_schema, vals, main_cols))
                        featurebuffer = FeatureBuffer(main_cols, vals)
                        w_df = pd.DataFrame(window_data, columns=['sort']+main_cols)
                        ins_dict = dict(zip(main_cols, transformer_data(main_schema, vals, main_cols)))

                        if func: # 自定义特征计算
                            func(featurebuffer, w_df, ins_dict)
                        if features: # 自动化配置特征计算
                            get_feature(featurebuffer, sort, w_df, ins_dict, features)
                        if down_schmea:
                            feat_out.write(output_sep.join(featurebuffer.get_feature_names()))
                            feat_out.write('\n')
                            down_schmea = False
                        feat_out.write(output_sep.join(str(x) for x in featurebuffer.get_feature_vals()))
                        feat_out.write('\n')
                        feat_out.flush()
                        # 存储已处理的样本
                        if log:
                            log_out.write('_'.join(vals))
                            log_out.write('\n')
                            log_out.flush()

                        end = time.time()
                        cnt += 1
                        logger.info(f'{cnt}th sample feature extract finished,  cost {round(end-start, 4)}s')

                logger.info(f'extract feature partition {i+1} data end')
        end_time = time.time()
        logger.info(f'task finished, cost {round(end_time-self.start_time, 4)}s')
        
        if log: # 删除日志文件
            os.remove(log_file)

    def map_partition_with_two_table(self, func):
        output_file, features = self.output_file, self.features
        main_cols, feat_cols = self.main_cols, self.feat_cols
        output_sep = self.output_sep

        log, log_data, write_flag = self.log, None, 'w'
        log_file = self.tmp_dir+f'/feat_extract_log.csv'
        if log:
            if os.path.exists(log_file):
                with open(log_file, 'r') as fin:
                    log_data = [f.strip() for f in fin]
            write_flag = 'a'
        
        down_schmea, cnt = True, 0
        with open(output_file, write_flag) as feat_out, open(log_file, write_flag) as log_out:
            for i in range(self.num_partitions):
                logger.info(f'extract feature partition {i+1} data start')
                tmp_file = self.tmp_dir+f'/{self.tmp_file_prefix}_num_partition_{i+1}_sorted'
                data = load_obj(tmp_file)
                for _, groups in itertools.groupby(data, key=lambda x: x[0]):
                    window_data = []
                    for _, sort, flag, vals in groups:
                        if flag == 0: # 特征
                            window_data.append([sort]+vals)
                        else: # 样本
                            if log_data and '_'.join(vals) in log_data: # 如果已经计算了该样本，直接跳过
                                down_schmea = False
                                continue
                            
                            start = time.time()
                            featurebuffer = FeatureBuffer(main_cols, vals)
                            w_df = pd.DataFrame(window_data, columns=[0]+feat_cols)
                            ins_dict = dict(zip(main_cols, vals))

                            if func: # 自定义特征计算
                                func(featurebuffer, pd.DataFrame(window_data, columns=[0]+feat_cols), ins_dict)
                            if features: # 自动化配置特征计算
                                get_feature(featurebuffer, sort, w_df, ins_dict, features)
                            if down_schmea:
                                feat_out.write(output_sep.join(featurebuffer.get_feature_names()))
                                feat_out.write('\n')
                                down_schmea = False
                            feat_out.write(output_sep.join(str(x) for x in featurebuffer.get_feature_vals()))
                            feat_out.write('\n')
                            feat_out.flush()
                            if log: # 存储已处理的样本
                                log_out.write('_'.join(vals))
                                log_out.write('\n')
                                log_out.flush()
                            
                            end = time.time()
                            cnt += 1
                            logger.info(f'{cnt}th sample feature extract finished,  cost {round(end-start, 4)}s')
                logger.info(f'extract feature partition {i+1} data end')
        
        end_time = time.time()
        logger.info(f'task finished, cost {round(end_time-self.start_time, 4)}s')

        if log: # 删除日志文件
            os.remove(log_file)
