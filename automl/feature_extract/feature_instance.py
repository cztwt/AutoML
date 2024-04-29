from collections import OrderedDict

class FeatureBuffer:
    '''特征存储器
    
    参数：
        - names: 特征名称
        - vals: 特征值
    '''
    def __init__(self, names, vals):
        self.features = OrderedDict()
        if names is not None:
            assert vals is not None and len(names) == len(vals), \
                    f'vals should not be None when names is not None, len(names) = {len(names)}, len(vals) = {len(vals)}'
            for name, value in zip(names, vals):
                self.features[name] = value
    
    def add_feature(self, name, val):
        '''添加特征
        
        参数：
            - name: 特征名称
            - val: 特征值
        '''
        if name in self.features:
            raise Exception(f'duplicated feature name {name}')
        self.features[name] = val
    
    def get_feature_names(self):
        return self.features.keys()
    
    def get_feature_vals(self):
        return self.features.values()
