import pandas as pd
from ast import literal_eval

def expand_dict_column(df, column_name):
    """将 DataFrame 中的包含字典字符串的列展开为多列，并与原始数据合并。

    参数：
        - df (DataFrame): 要处理的 DataFrame。
        - column_name (str): 要展开的包含字典字符串的列的名称。

    返回：
        - DataFrame: 包含展开后列的新 DataFrame。
    """
    if not isinstance(df, pd.DataFrame): raise Exception(f'parameter df should be padnas dataframe.')
    if isinstance(column_name, str):
        col_lst = [column_name]
    elif isinstance(column_name, list):
        col_lst = column_name
    else:
        raise Exception(f'parameter column_name should be str or list.')

    tmp_df = df.copy()
    res = pd.DataFrame()
    for col in col_lst:
        # 使用 ast.literal_eval 将字符串解析为字典对象
        tmp_df[col] = tmp_df[col].apply(literal_eval)
        # 使用 json_normalize 将字典列展开为多列
        expanded_df = pd.json_normalize(tmp_df[col])
        expanded_df.columns = [f"{col}_{key}" for key in expanded_df.columns]
        # 将展开的列与原始数据合并
        res = pd.concat([res, expanded_df], axis=1)
    result = pd.concat([tmp_df.drop(columns=col_lst, axis=1), res], axis=1)

    return result

def expand_list_column(df, column_name):
    """将 DataFrame 中的包含列表字符串的列展开为多列，并与原始数据合并。

    参数：
        - df (DataFrame): 要处理的 DataFrame。
        - column_name (str): 要展开的包含列表字符串的列的名称。

    返回：
        - DataFrame: 包含展开后列的新 DataFrame。
    """
    if not isinstance(df, pd.DataFrame): raise Exception(f'parameter df should be padnas dataframe.')
    if isinstance(column_name, str):
        col_lst = [column_name]
    elif isinstance(column_name, list):
        col_lst = column_name
    else:
        raise Exception(f'parameter column_name should be str or list.')
    
    tmp_df = df.copy()
    res = pd.DataFrame()
    for col in col_lst:
        tmp_df[col] = tmp_df[col].apply(literal_eval)
        expanded_df = pd.DataFrame(tmp_df[col].to_list())
        expanded_df.columns = [f"{col}_{val}" for val in expanded_df.columns]
        res = pd.concat([res, expanded_df], axis=1)
    result = pd.concat([tmp_df.drop(columns=col_lst, axis=1), expanded_df], axis=1)
    
    return result



if __name__ == '__main__':
    # expand_dict_column test
    df = pd.read_csv('/Users/chenzhao/Desktop/AutoML/automl/data_process/data_dict.txt', delimiter='|')
    result = expand_dict_column(df, 'feat2')
    print(result)
    # expand_list_column test
    df = pd.read_csv('/Users/chenzhao/Desktop/AutoML/automl/data_process/data_list.txt', delimiter='|')
    result = expand_list_column(df, 'feat2')
    print(result)

