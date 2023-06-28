import numpy as np
import collections
import random
import torch

# 定义迷宫环境
class MazeEnv:
    '''
    定义迷宫环境
    '''
    def __init__(self, state_lb, state_ub, x_terminal, y_terminal, epsilon):
        # 默认状态空间为正方形
        self.state_lb = state_lb # 状态空间的下界
        self.state_ub = state_ub # 状态空间的上界
        self.x = 0 # 记录智能体目前位置x
        self.y = 0 # 记录智能体目前位置y

        # 终点
        self.x_terminal = x_terminal
        self.y_terminal = y_terminal
        self.epsilon = epsilon # 距离终点的误差

    # 重置智能体位置
    def reset(self):
        # self.x = 0.0
        # self.y = 0.0
        self.x = np.random.uniform(self.state_lb, self.state_ub)
        self.y = np.random.uniform(self.state_lb, self.state_ub)

        return self.x, self.y

    # 环境对智能体的反馈
    def step(self, action, count):
        # 将行动解包
        x_action, y_action = action[0], action[1]
        # 计算行动后的位置
        # 行动后的位置不能超过边界
        self.x = min(self.state_ub, max(self.state_lb, self.x + x_action))
        self.y = min(self.state_ub, max(self.state_lb, self.y + y_action))
        next_state = (self.x, self.y)
        reward = -np.linalg.norm(np.array(next_state) - np.array([self.x_terminal, self.y_terminal]))
        done = False

        # 终止状态只有终点一种情况
        if np.linalg.norm(np.array([self.x, self.y]) - np.array([self.x_terminal, self.y_terminal])) < self.epsilon:
            done = True

        if count >= 500:
            done = True
            reward = -100

        return next_state, reward, done

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) # 在index=0处插入0，并计算累积值
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

class ReplayBuffer:
    '''
    经验回放池
    '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 采用队列存储

    def add(self, state, action, reward, next_state, done):  
        # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  
        # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)
