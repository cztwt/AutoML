from automl.util import *

def f_count(window_data):
    return len(window_data)

def f_distinct_count(window_data, val_col):
    return window_data[val_col].nunique()

def f_sum(window_data, val_col):
    return window_data[val_col].sum()

def f_avg(window_data, val_col):
    return window_data[val_col].mean()

def f_max(window_data, val_col):
    return window_data[val_col].max()

def f_min(window_data, val_col):
    return window_data[val_col].min()

def f_lag(window_data, cols, n=1):
    '''获取最近第n行的lag滞后数据
    
    场景：求最近一次(购买)的(金额)
    
    参数：
        - window_data: 已经按照对应的date列排序好的dataframe数据
        - cols: 需要获取的数据列名称,可以为字符串和list类型
        - n: 获取最近第几行的数据
    
    返回值：
        - res: list类型, 对应每个最近第n行的cols的具体值
    '''
    if isinstance(cols, str):
        col_lst = [s.strip() for s in cols.split(',')]
    elif isinstance(cols, list):
        col_lst = cols
    if len(window_data) == 0 or window_data is None:
        return [None]*len(col_lst)
    else:
        res = []
        for col in col_lst:
            res.append(window_data.iloc[-n][col])
        return res
    
def f_mon_avg(window_data, cols, ins_data):
    '''计算月日均：筛选样本时间所在[月份开始时间, 样本时间的数据)
    
    参数：
        - window_data: 已经按照对应的date列排序好的dataframe数据
        - cols: [date1, date2, val_col], 顺序不可乱, date1: 样本时间, date2 特征数据时间, val_col

    返回值：
        - res: 特征值
    '''
    if isinstance(cols, str):
        col_lst = [s.strip() for s in cols.split(',')]
    elif isinstance(cols, list):
        col_lst = cols
    if len(window_data) == 0 or window_data is None:
        return None
    else:
        ins_date = ins_data[col_lst[0]]
        year, month = ins_date.year, ins_date.month
        start_date = datetime.datetime(year, month, 1)
        data = window_data[(window_data[col_lst[1]] >= start_date) & (window_data[col_lst[1]] < ins_date)]
        days = (ins_date - start_date).days if len(data) != 0 else None
        res = data[col_lst[2]].sum() / days if days else None 
        return res

def f_quarter_avg(window_data, cols, ins_data):
    '''计算季日均：筛选样本时间所在[月份开始时间, 样本时间的数据)
    
    参数：
        - window_data: 已经按照对应的date列排序好的dataframe数据
        - cols: [date1, date2, val_col], 顺序不可乱, date1: 样本时间, date2 特征数据时间, val_col

    返回值：
        - res: 特征值
    '''
    if isinstance(cols, str):
        col_lst = [s.strip() for s in cols.split(',')]
    elif isinstance(cols, list):
        col_lst = cols
    if len(window_data) == 0 or window_data is None:
        return None
    else:
        quarter_starts = [(1, 1), (4, 1), (7, 1), (10, 1)]  # 每个季度的开始月份和日期
        ins_date, start_date = ins_data[col_lst[0]], None
        month = ins_date.month
        for start_month, start_day in quarter_starts:
            if month in range(start_month, start_month + 3):
                start_date = datetime.datetime(ins_date.year, start_month, start_day)
        
        data = window_data[(window_data[col_lst[1]] >= start_date) & (window_data[col_lst[1]] < ins_date)]
        days = (ins_date - start_date).days if len(data) != 0 else None
        res = data[col_lst[2]].sum() / days if days else None 
        return res

def f_year_avg(window_data, cols, ins_data):
    '''计算年日均：筛选样本时间所在[月份开始时间, 样本时间的数据)
    
    参数：
        - window_data: 已经按照对应的date列排序好的dataframe数据
        - cols: [date1, date2, val_col], 顺序不可乱, date1: 样本时间, date2 特征数据时间, val_col

    返回值：
        - res: 特征值
    '''
    if isinstance(cols, str):
        col_lst = [s.strip() for s in cols.split(',')]
    elif isinstance(cols, list):
        col_lst = cols
    if len(window_data) == 0 or window_data is None:
        return None
    else:
        ins_date = ins_data[col_lst[0]]
        start_date = datetime.datetime(ins_date.year, 1, 1)
        data = window_data[(window_data[col_lst[1]] >= start_date) & (window_data[col_lst[1]] < ins_date)]
        days = (ins_date - start_date).days if len(data) != 0 else None
        res = data[col_lst[2]].sum() / days if days else None 
        return res
    
def f_datediff(window_data, cols, ins_data):
    '''给定指定窗口数据统计样本时间距最近一次有数据的时间的diff

    场景：求最近一次购买时间距样本时间的天数

    参数：
        - window_data: 已经按照对应的date列排序好的dataframe数据
        - ins_data: 字典类型的样本数据, key表示列名称,value表示列名称对应的值
        - cols: [date1, date2], 顺序不可乱, date1: 样本时间, date2 特征数据时间
    
    返回值：
        - fea_vals: 特征值
    '''
    if isinstance(cols, str):
        col_lst = [s.strip() for s in cols.split(',')]
    elif isinstance(cols, list):
        col_lst = cols
    if len(window_data) == 0 or window_data is None:
        return None
    else:
        # ins_date = get_datetime(ins_data[col_lst[0]], col_lst[1])
        # fea_date = window_data.iloc[-1][col_lst[2]] if len(window_data) != 0 else None
        # fea_vals = (ins_date - get_datetime(fea_date, col_lst[3])).days if fea_date else None
        ins_date = ins_data[col_lst[0]]
        fea_date = window_data.iloc[-1][col_lst[1]] if len(window_data) != 0 else None
        fea_vals = (ins_date - fea_date).days if fea_date else None
        return fea_vals

def f_count_ratio(window_data, cols, ins_data):
    '''交叉维度特征计算
    
    场景：推荐中,求当前{产品号/产品风险等级/起购金额等级}历史{180/360}天申购{次数}的占比

    参数：
        - window_data: 已经按照对应的date列排序好的dataframe数据
        - ins_data: 字典类型的样本数据, key表示列名称,value表示列名称对应的值
        - cols: [ins_col, feat_col, calc_col], ins_col: 样本筛选字段名 feat_col: 特征筛选字段名 calc_col: 计算的列名

    返回值：
        - feat_vals: 特征值
    '''
    if isinstance(cols, str):
        col_lst = [s.strip() for s in cols.split(',')]
    elif isinstance(cols, list):
        col_lst = cols
    if len(window_data) == 0 or window_data is None:
        return None
    else:
        sel_count = len(window_data[window_data[col_lst[1]] == ins_data[col_lst[0]]])
        key_count = len(window_data)
        feat_vals = sel_count/key_count if key_count>0 else None
        return feat_vals

def f_sum_ratio(window_data, cols, ins_data):
    '''交叉维度特征计算
    
    场景：推荐中,求当前{产品号/产品风险等级/起购金额等级}历史{180/360}天申购{金额}的占比

    参数：
        - window_data: 已经按照对应的date列排序好的dataframe数据
        - ins_data: 字典类型的样本数据, key表示列名称,value表示列名称对应的值
        - cols: [ins_col, feat_col, calc_col], ins_col: 样本筛选字段名 feat_col: 特征筛选字段名 calc_col: 计算的列名

    返回值：
        - fea_val: 特征值
    '''
    if isinstance(cols, str):
        col_lst = [s.strip() for s in cols.split(',')]
    elif isinstance(cols, list):
        col_lst = cols
    if len(window_data) == 0 or window_data is None:
        return None
    else:
        sel_sum = window_data[window_data[col_lst[1]] == ins_data[col_lst[0]]][col_lst[2]].sum()
        key_sum = window_data[col_lst[2]].sum()
        fea_val = sel_sum/key_sum if key_sum>0 else None
        return fea_val