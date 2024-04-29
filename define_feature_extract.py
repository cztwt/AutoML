import pandas as pd
from dateutil.relativedelta import relativedelta

def define_feat_extract(featurebuffer, window_data, sample_data):
    '''根据窗口数据自定义提取特征的函数

    参数：
        - featurebuffer: 存储当前样本的特征数据的类
        - window_data: dataframe类型, 当前样本的历史窗口数据
        - sample_data: 字典, key为样本特征名称, value为样本特征值
    
    '''
    ############################特征计算逻辑-开始############################
    # 统计当前的历史top3的购买金额均值
    
    for w in [30, 60, 90, 180, 360]:
        start_date = sample_data['sample_date'] + relativedelta(days=-w)
        w_df = window_data[(window_data['trx_date'] >= start_date) & (window_data['trx_date'] < sample_data['sample_date'])]
        
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

