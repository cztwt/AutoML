from automl.feature_extract.feature_extractor import FeatureExtractor
from automl.util import *
from automl.models.classifier import *

import pandas as pd
import numpy as np
import datetime
import polars as pl
import itertools
import glob
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from dateutil.relativedelta import relativedelta

def generate_data(num_records, num_prd, cust_ids, start_date, end_date):
    # 随机选择 cust_id
    cust_selected = np.random.choice(cust_ids, num_records)

    # 生成 data_date 的时间范围
    date_range = pd.date_range(start_date, end_date)
    data_dates = np.random.choice(date_range, num_records, replace=True)

    # 生成 kind, cnt 和 val 的随机数据
    prds = [f'prd_{i}' for i in range(num_prd)]
    prods = np.random.choice(prds, num_records)
    kinds = np.random.choice([0, 1, 2, 3], num_records)
    vals = np.random.uniform(1, 1000, num_records)

    # 创建 DataFrame
    data = pd.DataFrame({
        'cust_id': cust_selected,
        'trx_date': data_dates,
        'prod_code': prods,
        'trx_code': kinds,
        'trx_amt': vals,
        'trx_shr': np.random.uniform(1, 1000, num_records)
    })

    # 随机生成正负样本
    # labels = np.random.choice([0, 1], num_records)
    # data['label'] == labels
    return data

# 生成特征数据集
# num_records = 1000000
# num_prd = 5
# cust_ids = ['cust_' + str(i) for i in range(0, 50)]
# start_date = pd.to_datetime('2022-08-07')
# end_date = pd.to_datetime('2023-08-01')
# data = generate_data(num_records, num_prd, cust_ids, start_date, end_date)
# data['timestamps'] = data['trx_date'].apply(lambda x: x.timestamp()*1000)
# data.to_csv('./automl/data/feat_data.csv', index=False)
# print(data.shape)

def generate_sample_data(num_cust, num_prd, start_date, end_date):
    '''生成样本数据集
    
    参数：
        - num_cust: 客户数
        - num_prd: 产品数
    '''
    custs = [f'cust_{i}' for i in range(num_cust)]
    prds = [f'prd_{i}' for i in range(num_prd)]

    # 笛卡尔积
    cust_prd = list(itertools.product(custs, prds))
    cust_prd_df = pd.DataFrame(cust_prd)
    cust_prd_df.columns = ['cust_id', 'prd_id']

    # 生成 sample_date 的时间范围
    date_range = pd.date_range(start_date, end_date)
    sample_dates = np.random.choice(date_range, num_cust * num_prd, replace=True)
    cust_prd_df['sample_date'] = sample_dates

    # 生成随机正负样本
    labels = np.random.choice([0, 1], num_cust*num_prd)
    cust_prd_df['label'] = labels

    return cust_prd_df

# 生成样本数据集
# num_cust = 50
# num_prd = 5
# sam_start_date = pd.to_datetime('2023-08-07')
# sam_end_date = pd.to_datetime('2023-08-10')
# sample_data = generate_sample_data(num_cust, num_prd, sam_start_date, sam_end_date)
# sample_data.to_csv('./automl/data/sample_table.csv', index=False)
# print(sample_data.shape)
# print(sample_data.head())

def define_feat_extract(featurebuffer, window_data, sample_data):
    '''根据窗口数据自定义提取特征的函数

    参数：
        - featurebuffer: 存储当前样本的特征数据的类
        - window_data: dataframe类型, 当前样本的历史窗口数据
        - sample_data: 字典, key为样本特征名称, value为样本特征值
    
    '''
    ############################特征计算逻辑-开始############################
    # 统计当前的历史top3的购买金额均值
    window_data['trx_date'] = pd.to_datetime(window_data['trx_date'], format='%Y-%m-%d')
    sample_date = datetime.datetime.strptime(sample_data['sample_date'], '%Y-%m-%d')
    
    for w in [30, 60, 90, 180, 360]:
        start_date = sample_date + relativedelta(days=-w)
        w_df = window_data[(window_data['trx_date'] >= start_date) & (window_data['trx_date'] < sample_date)]
        
        cond_data1 = w_df[w_df['trx_code'].isin([0, 1, 2])]
        cond_data2 = w_df[w_df['trx_code'].isin([3])]

        featurebuffer.add_feature(f'f_count(trx_amt)_w{w}d_cond1', len(cond_data1))
        featurebuffer.add_feature(f'f_count(trx_amt)_w{w}d_cond2', len(cond_data2))

        featurebuffer.add_feature(f'f_sum(trx_amt)_w{w}d_cond1', cond_data1['trx_amt'].sum())
        featurebuffer.add_feature(f'f_sum(trx_amt)_w{w}d_cond2', cond_data2['trx_amt'].sum())

        featurebuffer.add_feature(f'f_mean(trx_amt)_w{w}d_cond1', cond_data1['trx_amt'].mean())
        featurebuffer.add_feature(f'f_mean(trx_amt)_w{w}d_cond2', cond_data2['trx_amt'].mean())
    
    cond_data3 = w_df[w_df['trx_code'].isin([3])]
    featurebuffer.add_feature(f'f_count(trx_amt)_cond3', len(cond_data3))
    featurebuffer.add_feature(f'f_sum(trx_amt)_cond3', cond_data3['trx_amt'].sum())
    featurebuffer.add_feature(f'f_mean(trx_amt)_cond3', cond_data3['trx_amt'].mean())

    ############################特征计算逻辑-结束############################

# cython打包   

df = pd.read_csv('./automl/data/f_feat_data.csv')
print('特征个数为：', len(df.columns))
data = pd.read_csv('./automl/data/feat_data.csv')
data['trx_date'] = pd.to_datetime(data['trx_date'])
end_date = datetime.datetime.strptime('2023-08-07', '%Y-%m-%d')
lower_date = end_date + relativedelta(days=-0)
upper_date = end_date + relativedelta(days=-60)
# 1. 验证datediff函数
# print(df[['cust_id','prd_id','sample_date']+['f_datediff(sample_date_trx_date)_cond31']].head())
# feat_data = data[(data['cust_id'] == 'cust_0') & (data['trx_date'] < end_date) 
#                  & (data['trx_code'].isin([0, 1, 2]))].sort_values(by=['cust_id', 'trx_date'])
# print(end_date, feat_data.iloc[-1]['trx_date'], (end_date-feat_data.iloc[-1]['trx_date']).days)
# 2. 验证lag函数
# print(df[['cust_id','prd_id','sample_date']+['f_buy_lag1(trx_amt)_cond41']].head())
# feat_data = data[(data['cust_id'] == 'cust_0') & (data['trx_date'] < end_date) 
#                  & (data['trx_code'].isin([0, 1, 2]))].sort_values(by=['cust_id', 'trx_date'])
# print(feat_data.iloc[-1]['trx_amt'])
# 3. 没有筛选条件
# print(df[['cust_id','prd_id','sample_date']+['f_shuhui_sum(trx_amt)_w30d']].head())
# feat_data = data[(data['cust_id'] == 'cust_0') & (data['trx_date'] < lower_date) & (data['trx_date'] >=upper_date)]
# print(feat_data['trx_amt'].sum())
# 4. 窗口+筛选条件
print(df[['cust_id','prd_id','sample_date']+['f_buy_sum(trx_amt)_w60d_cond11']].head())
feat_data = data[(data['cust_id'] == 'cust_0') & (data['trx_date'] < lower_date) 
                 & (data['trx_date'] >=upper_date) & (data['trx_code'].isin([0, 1, 2]))]
print(feat_data['trx_amt'].sum())
exit()

# 特征个数为： 22
#   cust_id prd_id          sample_date  f_buy_sum(trx_amt)_w60d_cond11
# 0  cust_0  prd_0  2023-08-07 00:00:00                    1.112857e+06
# 1  cust_0  prd_2  2023-08-07 00:00:00                    1.112857e+06
# 2  cust_0  prd_1  2023-08-08 00:00:00                    1.088824e+06
# 3  cust_0  prd_4  2023-08-09 00:00:00                    1.060288e+06
# 4  cust_0  prd_3  2023-08-10 00:00:00                    1.045488e+06
# 1112856.8140360462


if __name__ == '__main__':
    FeatureExtractor(
        main_table_file='./automl/data/sample_table.csv',
        main_schema_file = './automl/data/base_schema.txt',
        main_partition_col='cust_id',
        main_sort_col='sample_date',
        
        mode='two',

        feat_table_file='./automl/data/feat_data.csv',
        feat_schema_file = './automl/data/feat_schema.txt',
        feat_partition_col='cust_id',
        feat_sort_col='trx_date',
        
        output_file='./automl/data/f_feat_data.csv',
        calc_config_file='./automl/data/feat_config.json',
        # tmp_dir='',
        # log=True
    # ).repartition(num_partitions=2).sort_by_partition().map_partition(define_feat_extract)
    ).repartition(num_partitions=2).sort_by_partition().map_partition()

# 自动化配置
# INFO 2024-04-16 21:48:08,588 (feature_extractor.py 371 map_partition_with_two_table) task finished, 250th samples, cost 47.4016s
# INFO 2024-04-16 21:51:05,999 (feature_extractor.py 371 map_partition_with_two_table) task finished, 250th samples, cost 43.1922s
# INFO 2024-04-16 22:08:39,696 (feature_extractor.py 371 map_partition_with_two_table) task finished, 250th samples, cost 41.3567s
# 自定义函数
# INFO 2024-04-16 21:45:44,468 (feature_extractor.py 371 map_partition_with_two_table) task finished, 250th samples, cost 42.9939s

# 47116 samples, 168w+ feature data, extract 97 features, cost 64min
# data = load_obj('./tmp_feature/feat_data_20240201_211535_num_partition_1_sorted')
# print(data[:10])


# print(pd.read_csv('./automl/data/sample_table.csv').info())
# with open('./automl/data/sample_table.csv', 'r') as f:
#     for line in f:
#         print(line.strip().split(','))
# print(pd.read_csv('./automl/data/sample_table.csv').shape[0])
# print(len(pd.read_csv('./automl/data/sample_table.csv')['cust_id'].unique()))
# print(pd.read_csv('./automl/data/feat_data.csv').shape[0]+pd.read_csv('./automl/data/sample_table.csv').shape[0])
# print(len(
#     set(
#         pd.read_csv('./automl/data/feat_data.csv')['cust_id'].unique()\
#         +pd.read_csv('./automl/data/sample_table.csv')['cust_id'].unique()
#     )
# ))
# file_paths = glob.glob('./tmp_feature/*.csv')
# print(file_paths)
# num = 0
# cust_set = set()
# for file in file_paths:
#     with open(file, 'r') as f:
#         for line in f:
#             num += 1
#             cust_set.add(line.strip().split(',')[0])
# print(num, len(cust_set))

