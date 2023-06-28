import numpy as np
import collections
import random
import torch

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
