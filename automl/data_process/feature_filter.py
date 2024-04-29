import numpy as np
from tqdm import tqdm

def feature_filter(train, test, id_, target):
    """特征过滤
    
    参数：
        - train: dataframe,训练集
        - test: dataframe,测试集
        - id_: list,不需要做训练的列
        - target: str,标签列
    
    返回值：
        - used_features: 经过过滤条件后的特征列
    """
    not_used = id_ + [target]

    used_features = test.describe().columns
    for col in tqdm(used_features):
        # test中全为Nan的特征
        if test.loc[test[col].isnull()].shape[0] == test.shape[0]:
            if col not in not_used:
                not_used += [col]

        # nunique为1的特征
        if train[col].nunique() == 1:
            if col not in not_used:
                not_used += [col]

        # test中的值都比train中的值要大(或小)的特征
        if test[col].min() > train[col].max() or test[col].max() < train[col].min():
            if col not in not_used:
                not_used += [col]

        # 包含inf的特征（数值溢出或无法收敛）
        if train.loc[train[col] == np.inf].shape[0] != 0 or test.loc[test[col] == np.inf].shape[0] != 0:
            not_used += [col]

    print(f"filtered features: {not_used}")
    used_features = [x for x in used_features if x not in not_used]
    return used_features


# if __name__ == '__main__':
#     import pandas as pd
#     from sklearn.model_selection import train_test_split
#     df = pd.read_csv('./automl/data/f_feat_data.csv')
#     X_train, X_test = train_test_split(df, test_size=0.3, random_state=42)
#     feats = feature_filter(X_train, X_test, ['cust_id','prd_id','sample_date'], 'label')
#     print(feats)