import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 损失函数
def loss_function(x):
    return x**2

# 梯度函数
def gradient(x):
    return 2*x

# 梯度下降算法
def gradient_descent(learning_rate=0.1, num_iterations=50):
    x = 4  # 初始值
    history = [x]
    for i in range(num_iterations):
        grad = gradient(x)
        x = x - learning_rate * grad
        history.append(x)
    return history

# 创建动画
fig, ax = plt.subplots()
line, = ax.plot([], [], 'r', label='Gradient Descent')
ax.set_xlim(-5, 5)
ax.set_ylim(0, 25)

def init():
    line.set_data([], [])
    return line,

def update(frame):
    x_values = np.linspace(-5, 5, 100)
    y_values = loss_function(x_values)
    ax.plot(x_values, y_values, 'b', label='Loss Function')
    ax.scatter(frame, loss_function(frame), color='red')
    line.set_data(frame, loss_function(frame))
    return line,

ani = FuncAnimation(fig, update, frames=gradient_descent(), init_func=init, blit=True)
ani.save('gradient_descent.gif', writer='imagemagick', fps=5)

plt.legend()
plt.show()
