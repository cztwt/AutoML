import re
import datetime
import numpy as np
import pandas as pd
from collections import OrderedDict
from automl.util import *
from automl.feature_extract.CONST import *
from automl.feature_extract.feature_func import *

def split_file_by_partition_col(input_file, extension_name, cols_name, sep, partition_col, sort_col, header, flag, num, output):
    '''将文件数据进行分片, 并使得同一个partition的数据在一个分片文件中
    
    参数：
        - input_file: 输入文件路径
        - extension_name: 文件后缀扩展名
        - cols_name: 表的列名称
        - sep: 文件分隔符
        - partition_col: 指定的分区列
        - sort_col: 指定排序的列
        - header: 第一行是否为列名
        - flag: 1样本 0特征
        - num: 分片文件数
        - output_dir: 输出文件路径
    '''
    first_header = header
    partition_col_idx, sort_col_idx = cols_name.index(partition_col), cols_name.index(sort_col)
    # 读取输入文件并按照 partition_col 进行哈希分片
    for line in txt_reader(input_file, extension_name=extension_name):
        line = line.strip()
        if line:  # 确保非空行
            if first_header:
                first_header = False
                continue
            else:
                tokens = line.split(sep)
                partition_val = tokens[partition_col_idx]
                sort_val = tokens[sort_col_idx]

                # 计算 partition_id 的哈希值，并获取对应的输出文件
                hash_value = get_partition(partition_val, num)
                output_file = output[hash_value]

                out_val = ','.join([partition_val, sort_val, str(flag)]) + '|' + line

                # 将数据写入输出文件中
                output_file.write(out_val + '\n')

def parse_schema(schema_path):
    schema = OrderedDict()
    with open(schema_path, 'r') as fin:
        for line in fin:
            field, field_type, *extra_params = line.strip().split(',')
            field, field_type = field.strip(), field_type.strip().upper()
            if field_type == 'DATETIME':
                if len(extra_params) == 0:
                    raise ValueError(f"datetime field need date fromat, like '%Y-%m-%d'")
                date_format = extra_params[0].strip()
                schema[field] = (field_type, date_format)
            else:
                schema[field] = field_type
    
    return schema

def transformer_data(schema, lst):
    res = []
    for field, val in zip(schema, lst):
        if val == '': res.append(np.nan)
        elif isinstance(schema[field], tuple):
            res.append(datetime.datetime.strptime(val, schema[field][1]))
        elif schema[field] == 'INT':
            res.append(int(val))
        elif schema[field] == 'FLOAT':
            res.append(float(val))
        else:
            res.append(val)
    return res

def check_cols_type(var):
    '''计算列名类型检查'''
    if isinstance(var, list):
        return var
    elif isinstance(var, str):
        return [s.strip() for s in var.split(',')]
    else:
        raise TypeError(f'Unsupported type {type(var)}, should be list or str, the sep for str type is ,')

def check_cond_type(var):
    '''计算列名类型检查'''
    if isinstance(var, str):
        return [s.strip() for s in var.split('|')]
    return var

def parse_vals(var, sep=','):
    if isinstance(var, str):
        return [s.strip() for s in var.split(sep)]
    return var

def parse_window(var):
    def check(w_lst):
        res = {'val': [], 'interval': [], 'date': [], 'name': []}
        for w in w_lst:
            w_r = w.replace('，', ',')
            interval = ''.join([s for s in w_r if s in ['[', ']', '(', ')']])
            if not interval in WINDOW_INTERVAL:
                raise ValueError(f"Unsupported interval is {interval}, should be '[]','()','[)','(]', ''")
            if interval == '':
                num, let = parse_string(w_r)
                w_r = f'[{num}{let}, 0{let})' if let else f'[{num}, 0)'
                interval = '[)'
            upper, lower = [parse_string(s) for s in w_r.split(',')]
            res['interval'].append(interval)
            res['date'].append((upper[1], lower[1]))
            if upper[1] and lower[1]:
                res['val'].append((get_window_range(upper[0], upper[1]), get_window_range(lower[0], lower[1])))
            else:
                res['val'].append((upper[0], lower[0]))
            res['name'].append(set_window_name(upper, lower)) # 窗口特征名称 w60d w180d_60d
        return res
    if isinstance(var, list):
        return check(var)
    elif isinstance(var, str):
        w_lst = [s.strip() for s in var.split('|')]
        return check(w_lst)

def parse_calc_config(configs):
    '''解析配置文件'''
    res = []
    for config in configs:
        stats = config.get('stats')
        default_cols = config.get("default_cols")
        default_window = config.get("default_window")
        default_condition = config.get("default_condition")
        if isinstance(stats, list):
            d_cols = check_cols_type(default_cols)
            d_window = parse_window(default_window)
            d_condition = parse_vals(default_condition, '|')
            
            for stat in stats:
                res_config = {
                    "id": config["id"],
                    "name": config["name"],
                    "feat_prefix": config["feat_prefix"],
                    "stats": {
                        stat: {
                            "cols": d_cols,
                            "window": d_window,
                            "condition": d_condition
                        }
                    }
                }
                res.append(res_config)
        elif isinstance(stats, dict):
            res_config = {
                "id": config["id"],
                "name": config["name"],
                "feat_prefix": config["feat_prefix"],
                "stats": {}
            }
            for stat, stat_config in stats.items():
                stats_dict = {}
                cols = stat_config.get('cols')
                window = stat_config.get('window')
                condition = stat_config.get('condition')
                # 计算列解析
                if cols is not None:
                    stats_dict['cols'] = parse_vals(cols)
                elif default_cols is not None:
                    stats_dict['cols'] = parse_vals(default_cols)
                else:
                    raise ValueError(f'default_cols and cols are list types with non zero length')
                # 时间窗口解析
                if window is not None:
                    stats_dict['window'] = parse_window(window)
                elif default_window is not None:
                    stats_dict['window'] = parse_window(default_window)
                # 条件表达式解析
                if condition is not None:
                    stats_dict['condition'] = parse_vals(condition, '|')
                elif default_condition is not None:
                    stats_dict['condition'] = parse_vals(default_condition, '|')
                
                res_config['stats'][stat] = stats_dict
            res.append(res_config)
        else:
            raise TypeError(f'Unsupported type stats is {type(stats)}, should be list or dict')
    return res

# def parse_calc_config(configs):
#     '''解析配置文件'''
#     res = []
#     for config in configs:
#         stats = config.get('stats')
#         if isinstance(stats, list):
#             res_config = {
#                 "id": config["id"],
#                 "name": config["name"],
#                 "feat_prefix": config["feat_prefix"],
#                 "stats": {}
#             }
#             cols = check_cols_type(config.get("default_cols"))
#             window = parse_window(config.get("default_window"))
#             condition = parse_vals(config.get("default_condition"), '|')
#             group = config.get("default_group")
#             for stat in config["stats"]:
#                 res_config["stats"][stat] = {
#                     "cols": cols,
#                     "window": window,
#                     "condition": condition,
#                     "group": group,
#                 }
#             res.append(res_config)
#         elif isinstance(stats, dict):
#             res_config = {
#                 "id": config["id"],
#                 "name": config["name"],
#                 "feat_prefix": config["feat_prefix"],
#                 "stats": {}
#             }
#             d_cols = config.get("default_cols")
#             d_window = config.get('default_window')
#             d_condition = config.get('default_condition')
#             d_group = config.get('default_group')
#             for stat in config['stats']:
#                 stats_dict = {}
#                 # 计算列解析
#                 cols = config['stats'][stat].get('cols')
#                 if cols is not None:
#                     stats_dict['cols'] = parse_vals(cols)
#                 elif d_cols is not None:
#                     stats_dict['cols'] = parse_vals(d_cols)
#                 else:
#                     raise ValueError(f'default_cols and cols are list types with non zero length')
#                 # 时间窗口解析
#                 window = config['stats'][stat].get('window')
#                 if window is not None:
#                     stats_dict['window'] = parse_window(window)
#                 elif d_window is not None:
#                     stats_dict['window'] = parse_window(d_window)
#                 # 条件表达式解析
#                 condition = config['stats'][stat].get('condition')
#                 if condition is not None:
#                     stats_dict['condition'] = parse_vals(condition, '|')
#                 elif d_condition is not None:
#                     stats_dict['condition'] = parse_vals(d_condition, '|')
                
#                 res_config['stats'][stat] = stats_dict
#             res.append(res_config)
#         else:
#             raise TypeError(f'Unsupported type stats is {type(stats)}, should be list or dict')
#     return res

def trans_config(confis):
    res_config = OrderedDict()
    
    for config in confis:
        id = config.get('id')
        feat_prefix = config.get('feat_prefix')
        stats = config.get('stats')

        for stat in stats:
            cols = stats[stat].get('cols')
            window = stats[stat].get('window')
            condition = stats[stat].get('condition')

            for w, i, d, name in zip(window['val'], window['interval'], window['date'], window['name']) if window else [(None, None, None, None)]:
                for j, cond in enumerate(condition) if condition else [(0, None)]:
                    for col in cols:
                        feat_name = set_feat_name(feat_prefix, stat, col, window, name, cond, id, j)
                        feat_vals = [stat, col, cond, (w, i, d)]
                        if feat_name in res_config.keys():
                            raise Exception(f'duplicated feature name {feat_name}')
                        res_config[feat_name] = feat_vals

    grouped_data = OrderedDict()
    for key, value in res_config.items():
        window, cond = value[3], value[2]
        if cond not in grouped_data:
            grouped_data[cond] = OrderedDict()
        if window not in grouped_data[cond]:
            grouped_data[cond][window] = []
        grouped_data[cond][window].append((key, value))

    return grouped_data

def set_feat_name(prefix, stat, col, w, w_name, cond, id, j):
    if stat in ['count_ratio', 'sum_ratio']:
        col_str = '_'.join(col)
    elif stat == 'datediff':
        col_str = '_'.join([col[0], col[1]])
    else:
        col_str = col
    w_str = f'_{w_name}' if w else ''
    cond_str = f'_cond{id}{j+1}' if cond else ''
    return f"{prefix}_{stat}({col_str}){w_str}{cond_str}"


def check_key(dict, keys):
    '''检查特征计算配置文件中的key是否为指定的key'''
    for key in dict.keys():
        if key.strip().upper() not in keys:
            raise KeyError(f"unexpected key '{key}' found in the dictionary.")

def check_data_type(schema_file, vals):
    with open(schema_file, 'r') as fin:
        for line in fin:
            data_type = line.split(',')[1].strip()
            if data_type.upper() not in vals:
                raise TypeError(f'unexpected data type {data_type}')

def parse_string(string):
    numbers = re.findall(r'\d+', string)
    letters = re.findall(r'[a-zA-Z]+', string)
    return int(numbers[0]), letters[0].lower() if len(letters) != 0 else None

def set_window_name(u, l):
    if u[1] and l[1]:
        return f'w{u[0]}{u[1]}' if l[0] == 0 else f'w{u[0]}{u[1]}_{l[0]}{l[1]}'
    else:
        return f'r{u[0]}' if l[0] == 0 else f'r{u[0]}_{l[0]}'

def get_window_range(num, format):
    if format == 'd':
        return 86400 * num * 1000
    elif format == 'h':
        return 3600 * num * 1000
    elif format == 'm':
        return  60 * num * 1000
    elif format == 's':
        return num * 1000
    elif format == 'ms':
        return  num

def get_window_data(df, ins_timestamp, window):
    w, i, d = window[0], window[1], window[2]
    if d[0] and d[1]: # 按时间滑动窗口数据
        upper_date, lower_date = ins_timestamp-w[0], ins_timestamp-w[1]
        if i == '[]':
            w_data = df[(df.iloc[:,0] >= upper_date) & (df.iloc[:,0] <= lower_date)]
        elif i == '[)':
            w_data = df[(df.iloc[:,0] >= upper_date) & (df.iloc[:,0] < lower_date)]
        elif i == '(]':
            w_data = df[(df.iloc[:,0] > upper_date) & (df.iloc[:,0] <= lower_date)]
        elif i == '()':
            w_data = df[(df.iloc[:,0] > upper_date) & (df.iloc[:,0] < lower_date)]
    else: # 按条数滑动窗口数据
        # [10, 1] [10, 1) (10, 1] (10, 1) 1
        l = -1 if w[1] == 0 else -w[1]
        if w[0] == 1:
            w_data = df.iloc[-w[0]:]
        else:
            if i == '[]':
                w_data = pd.concat([df.iloc[-w[0]:l], df.iloc[l].to_frame().T])
            elif i == '[)':
                w_data = df.iloc[-w[0]:l]
            elif i == '(]':
                w_data = pd.concat([df.iloc[-w[0]+1:l], df.iloc[l].to_frame().T])
            elif i == '()':
                w_data = df.iloc[-w[0]+1:l]
    return w_data

def get_condition_data(df, cond_str, cache):
    cond_data = cache[cond_str]
    if cond_data is None:
        cond_data = df.query(cond_str)
        cache[cond_str] = cond_data
    return cond_data

def get_feature_val(df, col, ins, stat):
    if stat == 'sum':
        return f_sum(df, col)
    elif stat == 'count':
        return f_count(df)
    elif stat == 'distinct_count':
        return f_distinct_count(df, col)
    elif stat == 'avg':
        return f_avg(df, col)
    elif stat == 'max':
        return f_max(df, col)
    elif stat == 'min':
        return f_min(df, col)
    elif stat == 'mon_avg':
        return f_mon_avg(df, col, ins)
    elif stat == 'quarter_avg':
        return f_quarter_avg(df, col, ins)
    elif stat == 'year_avg':
        return f_year_avg(df, col, ins)
    elif stat == 'count_ratio':
        return f_count_ratio(df, col, ins)
    elif stat == 'sum_ratio':
        return f_sum_ratio(df, col, ins)
    elif stat == 'datediff':
        return f_datediff(df, col, ins)
    elif stat[:3] == 'lag':
        return f_lag(df, col, int(stat[3:]))[0]
    else:
        raise ValueError(f'unexpected stat: {stat}')

def get_feature(ins, ins_timestamp, df, ins_dict, feats):
    for cond_str, vals in feats.items():
        # 筛选条件数据
        cond_data = df.query(cond_str) if cond_str else df
        for window, v in vals.items():
            # 筛选窗口数据
            w_cond_data = get_window_data(cond_data, ins_timestamp, window) if window[0] else cond_data
            for feat_name, feat_vals in v:
                # 计算特征值
                feat_val = get_feature_val(w_cond_data, feat_vals[1], ins_dict, feat_vals[0])
                # 添加特征名称和特征值
                ins.add_feature(feat_name, feat_val)