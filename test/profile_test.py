from automl.feature_extract.feature_extractor import FeatureExtractor
from automl.util import *
from automl.models.classifier import *

import pandas as pd
import numpy as np

from dateutil.relativedelta import relativedelta
import cProfile
import pstats

def main():
    FeatureExtractor(
        main_table_file='./automl/data/sample_table.csv',
        main_schema_file = './automl/data/base_schema.txt',
        # main_table_file_extension='.csv',
        # main_table_sep=',',
        main_partition_col='cust_id',
        main_sort_col='sample_date',
        main_sort_col_format='%Y-%m-%d',
        
        mode='two',

        feat_table_file='./automl/data/feat_data.csv',
        feat_schema_file = './automl/data/feat_schema.txt',
        # feat_table_file_extension='.csv',
        # feat_table_sep=',',
        feat_partition_col='cust_id',
        feat_sort_col='trx_date',
        feat_sort_col_format='%Y-%m-%d',
        
        output_file='./automl/data/f_feat_data.csv',
        calc_config_file='./automl/data/feat_config.json',
        # tmp_dir='',
        # log=True
    # ).repartition(num_partitions=2).sort_by_partition().map_partition(define_feat_extract)
    ).repartition(num_partitions=2).sort_by_partition().map_partition()

if __name__ == '__main__':
    with cProfile.Profile() as profile:
    # cProfile.run('main()', sort='cumtime')
        main()
    stats = pstats.Stats(profile).sort_stats(pstats.SortKey.TIME)
    stats.print_stats(0.5)
    stats.dump_stats('./stats.txt')
