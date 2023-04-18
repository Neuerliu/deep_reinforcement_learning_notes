import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # 显示循环进度条的库


class CliffWalkingEnv:
    '''
    在Sarsa算法下的悬崖漫步环境
    '''
    def __init__(self, ncol, nrow):
        self.nrow = nrow
        self.ncol = ncol
        self.x = 0  # 记录当前智能体位置的横坐标，初始位置在左下角
        self.y = self.nrow - 1  # 记录当前智能体位置的纵坐标，初始位置在左下角

    def step(self, action):
        '''
        智能体采用action，返回此时的奖励和下一个状态
        '''
        # 4种动作, change[0]:上, change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0])) # 最大x不能超过self.ncol-1
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1])) # 最大y不能超过self.nrow-1
        next_state = self.y * self.ncol + self.x # 下一个状态
        reward = -1
        done = False

        # 判断是否为终止状态以及终止状态下的奖励
        if self.y == self.nrow - 1 and self.x > 0:  # 下一个位置在悬崖或者目标
            done = True
            # 如果在悬崖
            if self.x != self.ncol - 1:
                reward = -100
                
        return next_state, reward, done

    def reset(self):
        '''
        回归初始状态,坐标轴原点在左上角
        '''
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x
