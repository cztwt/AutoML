
from fefuncs import *
import numpy as np
import pandas as pd
import time

if __name__ == '__main__':
    df = pd.DataFrame({
        'feat1': np.random.randint(0, 100, 10000000),
        'feat2': np.random.randint(0, 100, 10000000)
    })
    data = list(df['feat1'])
    print(type(data), len(data))

    start = time.time()
    sum(data)
    print(f'python code cost: {round(time.time() - start, 4)}s')

    start = time.time()
    f_sum(data)
    print(f'cython code cost: {round(time.time() - start, 4)}s')

    start = time.time()
    np.sum(data)
    print(f'numpy list data  code cost: {round(time.time() - start, 4)}s')

    start = time.time()
    np.sum(df['feat1'])
    print(f'numpy dataframe data  code cost: {round(time.time() - start, 4)}s')

    start = time.time()
    df['feat1'].sum()
    print(f'pandas code cost: {round(time.time() - start, 4)}s')
