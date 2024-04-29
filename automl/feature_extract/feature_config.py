import re
from collections import OrderedDict

# 窗口的区间配置
WINDOW_INTERVAL = ['[]','()','[)','(]','']
# 表配置需要的字段
TABLE_KEYS = ['table', 'partition_col', 'sort_col', 'schema', 'file_path', 'header', 'sep', 'extension']

class FeatureConfigHelper:
    '''特征衍生配置文件解析类
    
    参数：
        - conf_file_path: yaml配置文件路径
    '''
    def __init__(self, conf):
        self.conf = conf

    def get_feature_extract_params(self):
        conf = self.conf
        table_num, main_table_name, table_lst = self.get_tables()
        features = self.get_features()
        params = {}
        main_table = next(item for item in table_lst if item.get('table') == main_table_name)
        params['main_table_file'] = main_table['file_path']
        params['main_schema'] = main_table['schema']
        params['main_partition_col'] = main_table['partition_col']
        params['main_sort_col'] = main_table['sort_col']
        params['main_table_sep'] = main_table.get('sep', ',')
        params['main_header'] = main_table.get('header', True)
        params['main_table_file_extension_name'] = main_table.get('extension', '.csv')
        if table_num > 1:
            feat_table = next(item for item in table_lst if item.get('table') != main_table_name)
            params['feat_table_file'] = feat_table['file_path']
            params['feat_schema'] = feat_table['schema']
            params['feat_partition_col'] = feat_table['partition_col']
            params['feat_sort_col'] = feat_table['sort_col']
            params['feat_table_sep'] = feat_table.get('sep', ',')
            params['feat_header'] = feat_table.get('header', True)
            params['feat_table_file_extension_name'] = feat_table.get('extension', '.csv')
        
        params['mode'] = table_num
        params['output_file'] = conf['output']['file_path']
        params['output_sep'] = conf['output'].get('sep', ',')
        params['features'] = features
        params['log'] = conf.get('log', False)
        params['tmp_dir'] = conf.get('tmp_dir', './tmp_feature')
        return params
    
    def get_schema(self, schema_str):
        schema = OrderedDict()
        for s in schema_str.split(','):
            field, field_type, *extra_params = s.split('|')
            field, field_type = field.strip(), field_type.strip().upper()
            if field_type == 'DATETIME':
                if len(extra_params) == 0:
                    raise ValueError(f"datetime field need date fromat, like '%Y-%m-%d'")
                date_format = extra_params[0].strip()
                schema[field] = (field_type, date_format)
            else:
                schema[field] = field_type
        return schema
    
    def _parse_table(self, table):

        for key in table.keys():
            if key not in TABLE_KEYS:
                raise Exception(f'unsupperted table key: {key}')
        
        table_name = table.get('table')
        if not table_name:
            raise Exception('table name should not be None.')
        if not table.get('partition_col'):
            raise Exception(f'{table_name} table partition col should not be None.')
        if not table.get('sort_col'):
            raise Exception(f'{table_name} table sort col should not be None.')
        if not table.get('schema'):
            raise Exception(f'{table_name} table schema should not be None.')
        if not table.get('file_path'):
            raise Exception(f'{table_name} table file path should not be None.')

        return {
            'table': table['table'],
            'partition_col': table['partition_col'],
            'sort_col': table['sort_col'],
            'schema': self.get_schema(table['schema']),
            'file_path': table['file_path']
        }
        
    def get_tables(self):
        tables, main_table = self.conf['tables'], self.conf['main_table']
        table_num = len(tables)
        if table_num not in [1, 2]:
            raise Exception(f"unsupported table num: {len(tables)}, expected one or two table")
        
        table_lst = [self._parse_table(table) for table in tables]
        return [table_num, main_table, table_lst]
    
    def get_features(self):
        if not self.conf.get('features'): return None
        res_config = OrderedDict()
        features = self.parse_features(self.conf['features'])
    
        for config in features:
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
                            feat_name = self.set_feat_name(feat_prefix, stat, col, window, name, cond, id, j)
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

    def parse_features(self, configs):
        '''解析配置文件'''
        res = []
        for config in configs:
            stats = config.get('stats')
            default_cols = config.get("default_cols")
            default_window = config.get("default_window")
            default_condition = config.get("default_condition")
            if isinstance(stats, list):
                d_cols = self.check_cols_type(default_cols)
                d_window = self.parse_window(default_window)
                d_condition = self.parse_vals(default_condition, '|')
                
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
                        stats_dict['cols'] = self.parse_vals(cols)
                    elif default_cols is not None:
                        stats_dict['cols'] = self.parse_vals(default_cols)
                    else:
                        raise ValueError(f'default_cols and cols are list types with non zero length')
                    # 时间窗口解析
                    if window is not None:
                        stats_dict['window'] = self.parse_window(window)
                    elif default_window is not None:
                        stats_dict['window'] = self.parse_window(default_window)
                    # 条件表达式解析
                    if condition is not None:
                        stats_dict['condition'] = self.parse_vals(condition, '|')
                    elif default_condition is not None:
                        stats_dict['condition'] = self.parse_vals(default_condition, '|')
                    
                    res_config['stats'][stat] = stats_dict
                res.append(res_config)
            else:
                raise TypeError(f'Unsupported type stats is {type(stats)}, should be list or dict')
        return res
    
    def set_feat_name(self, prefix, stat, col, w, w_name, cond, id, j):
        if stat in ['count_ratio', 'sum_ratio']:
            col_str = '_'.join(col)
        elif stat == 'datediff':
            col_str = '_'.join([col[0], col[1]])
        else:
            col_str = col
        w_str = f'_{w_name}' if w else ''
        cond_str = f'_cond{id}{j+1}' if cond else ''
        return f"{prefix}_{stat}({col_str}){w_str}{cond_str}"
    
    def check_cols_type(self, var):
        '''计算列名类型检查'''
        if isinstance(var, list):
            return var
        elif isinstance(var, str):
            return [s.strip() for s in var.split(',')]
        else:
            raise TypeError(f'Unsupported type {type(var)}, should be list or str, the sep for str type is ,')

    def check_cond_type(self, var):
        '''计算列名类型检查'''
        if isinstance(var, str):
            return [s.strip() for s in var.split('|')]
        return var

    def parse_vals(self, var, sep=','):
        if isinstance(var, str):
            return [s.strip() for s in var.split(sep)]
        return var

    def parse_window(self, var):
        def check(w_lst):
            res = {'val': [], 'interval': [], 'date': [], 'name': []}
            for w in w_lst:
                w_r = w.replace('，', ',')
                interval = ''.join([s for s in w_r if s in ['[', ']', '(', ')']])
                if not interval in WINDOW_INTERVAL:
                    raise ValueError(f"Unsupported interval is {interval}, should be '[]','()','[)','(]', ''")
                if interval == '':
                    num, let = self.parse_string(w_r)
                    w_r = f'[{num}{let}, 0{let})' if let else f'[{num}, 0)'
                    interval = '[)'
                upper, lower = [self.parse_string(s) for s in w_r.split(',')]
                res['interval'].append(interval)
                res['date'].append((upper[1], lower[1]))
                if upper[1] and lower[1]:
                    res['val'].append((self.get_window_range(upper[0], upper[1]), self.get_window_range(lower[0], lower[1])))
                else:
                    res['val'].append((upper[0], lower[0]))
                res['name'].append(self.set_window_name(upper, lower)) # 窗口特征名称 w60d w180d_60d
            return res
        if isinstance(var, list):
            return check(var)
        elif isinstance(var, str):
            w_lst = [s.strip() for s in var.split('|')]
            return check(w_lst)
    
    def parse_string(self, string):
        numbers = re.findall(r'\d+', string)
        letters = re.findall(r'[a-zA-Z]+', string)
        return int(numbers[0]), letters[0].lower() if len(letters) != 0 else None

    def set_window_name(self, u, l):
        if u[1] and l[1]:
            return f'w{u[0]}{u[1]}' if l[0] == 0 else f'w{u[0]}{u[1]}_{l[0]}{l[1]}'
        else:
            return f'r{u[0]}' if l[0] == 0 else f'r{u[0]}_{l[0]}'

    def get_window_range(self, num, format):
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

        
        
        
    

    

    


if __name__ == '__main__':
    import yaml
    conf_file = '/Users/chenzhao/Desktop/AutoML/conf_test/conf.yaml'
    with open(conf_file, 'r') as fin:
        yaml_reader = yaml.load(fin, Loader=yaml.FullLoader)
    fch = FeatureConfigHelper(yaml_reader)
    print(fch.conf)
    # 获取版本号测试
    print(fch.get_version())
    # schema
    print(fch.get_schema(yaml_reader['tables'][0]['schema']))
    # tables
    print('='*40+'tables'+'='*40)
    print(fch.get_tables())
    # features
    print('='*40+'features'+'='*40)
    features = fch.get_features()
    for cond, vals in features.items():
        print(cond)
        print('*'*40)
        for k, v in vals.items():
            print(k)
            print(v)
        print('='*40)