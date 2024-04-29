import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from automl.util import *

from sklearn.linear_model import LinearRegression

# 使用线性回归模型拟合数据
# h_df = pd.read_csv('/Users/chenzhao/LearnDir/ml/liner_regression/kc_house_data_NaN.csv')
# model = LinearRegression()
# X, y = h_df['sqft_living'].values.reshape(-1,1), h_df['price']
# model.fit(X, y)

# # 打印模型的斜率和截距
# print("斜率(Coefficient): ", model.coef_)
# print("截距(Intercept): ", model.intercept_)

# # 绘制数据点
# plt.scatter(X, y, color='#4BAEEB', alpha=0.5)
# # 绘制拟合直线
# plt.plot(X, model.predict(X), color='red', linewidth=2)
# # 获取当前的坐标轴对象
# ax = plt.gca()
# # 禁用科学计数法
# ax.get_xaxis().get_major_formatter().set_scientific(False)
# ax.get_yaxis().get_major_formatter().set_scientific(False)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# plt.xlabel('X(size of living area)')
# plt.ylabel('y(price)')
# plt.savefig('my_plot.png')
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt

# # 生成x轴上的数据点
# x = np.linspace(0, 15, 20)

# # 使用y = 2*x + 4 生成对应的y轴上的数据点
# y = 2*x + 4

# # 加入一些随机噪声
# y_noise = y + np.random.normal(1, 3, size=x.shape)

# # 计算拟合直线的斜率和截距
# slope, intercept = np.polyfit(x, y_noise, 1)

# # 计算每个点到直线的垂直距离
# distances = np.abs(slope*x - y_noise + intercept) / np.sqrt(slope**2 + 1)
# # 画出数据点和拟合直线
# plt.scatter(x, y_noise, label='Data')
# plt.plot(x, slope*x + intercept, color='r', label='Fitted line: y = {:.2f}x + {:.2f}'.format(slope, intercept))
# # 画出距禽直线的距离
# plt.vlines(x, y_noise, slope*x + intercept, colors='g', linestyles='dotted', label='Distances')
# # 获取当前的坐标轴对象
# ax = plt.gca()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# # 设置坐标轴标签
# plt.xlabel('X(size of living area)')
# plt.ylabel('y(price)')
# # 去掉x轴和y轴的刻度
# plt.xticks([])
# plt.yticks([])
# # 显示图像
# plt.show()





# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # 定义要优化的函数（这里以一个简单的二次函数为例）
# def func(x, y):
#     return x**2 + y**2

# # 定义梯度函数
# def gradient(x, y):
#     return np.array([2*x, 2*y])

# # 梯度下降算法
# def gradient_descent(learning_rate, steps):
#     path = []
#     x, y = 4, 4  # 初始点
#     for _ in range(steps):
#         path.append((x, y))
#         grad = gradient(x, y)
#         x -= learning_rate * grad[0]
#         y -= learning_rate * grad[1]
    
#     return path

# # 设置梯度下降参数
# learning_rate = 0.1
# steps = 20

# # 运行梯度下降算法
# path = gradient_descent(learning_rate, steps)

# # 绘制动图
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # 生成网格数据
# X = np.linspace(-5, 5, 100)
# Y = np.linspace(-5, 5, 100)
# X, Y = np.meshgrid(X, Y)
# Z = func(X, Y)

# # 绘制曲面
# ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

# # 绘制优化路径
# xs, ys = zip(*path)
# zs = [func(x, y) for x, y in path]
# ax.plot(xs, ys, zs, marker='o', color='r')

# plt.show()

from functools import lru_cache

# @lru_cache
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

start_time = time.time()
print(fibonacci(35))
print(f'{time.time()-start_time}')

@lru_cache
def fibonacci2(n):
    if n < 2:
        return n
    return fibonacci2(n-1) + fibonacci2(n-2)

start_time = time.time()
print(fibonacci2(35))
print(f'{time.time()-start_time}')