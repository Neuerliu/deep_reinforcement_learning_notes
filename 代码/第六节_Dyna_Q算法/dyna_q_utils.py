import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import time

class CliffWalkingEnv:
    '''
    定义Dyna-Q的悬崖漫步环境
    '''
    def __init__(self, ncol, nrow):
        # 行列初始化
        self.nrow = nrow
        self.ncol = ncol
        self.x = 0 # 当前智能体的横坐标
        self.y = self.nrow - 1 # 当前智能体的纵坐标

    # 环境和智能体交互函数
    def step(self, action):
        # 4种动作, change[0]:上, change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        # 定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0])) # 不能超过横坐标限制
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1])) # 不能超过纵坐标限制
        next_state = self.y * self.ncol + self.x # 下一个状态
        reward = -1
        done = False
        if self.y == self.nrow - 1 and self.x > 0:  # 下一个位置在悬崖或者目标
            done = True
            # 下一个状态在悬崖
            if self.x != self.ncol - 1:
                reward = -100
        
        return next_state, reward, done

    # 重置函数
    def reset(self):  # 回归初始状态,起点在左上角
        self.x = 0
        self.y = self.nrow - 1

        return self.y * self.ncol + self.x

# 定义dyna-Q智能体
class DynaQ:
    '''
    Dyna-Q算法

    输入:
    :param ncol: 列数
    :param nrow: 行数
    :param epsilon: epsilon-greedy中的参数
    :param alpha: 学习率
    :param gamma: 折扣因子
    :param n_planning: 使用模型模拟数据更新Q函数的次数
    :param n_action: 动作个数
    '''
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_planning, n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action]) # 初始化动作价值函数
        self.n_action = n_action # 动作个数
        self.alpha = alpha # 学习率
        self.gamma = gamma # 折扣因子
        self.epsilon = epsilon # epsilon-greedy中的参数

        self.n_planning = n_planning # 使用模型模拟数据更新Q函数的次数，对应一次Q-learning
        self.model = dict() # 初始化模型

    # 采用epsilon-greedy策略选择动作
    def take_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action
    
    # 使用采样得到的真实数据更新Q函数
    def q_learning(self, s0, a0, r, s1):
        td_error = r + self.gamma * np.max(self.Q_table[s1]) - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error

    # 完整的更新动作价值函数
    def update(self, s0, a0, r, s1):
        self.q_learning(s0, a0, r, s1)
        self.model[(s0, a0)] = r, s1 # 利用采样数据更新环境模型
        for _ in range(self.n_planning):
            # 从环境的字典中随机选择曾经遇到的状态-动作对
            (s, a), (r, s_) = random.choice(list(self.model.items()))
            self.q_learning(s, a, r, s_) # 利用模型模拟数据更新Q函数

# 定义dyna-Q的训练函数
def DynaQ_CliffWalking(n_planning):
    ncol = 12
    nrow = 4
    env = CliffWalkingEnv(ncol, nrow)
    epsilon = 0.01
    alpha = 0.1
    gamma = 0.9
    agent = DynaQ(ncol, nrow, epsilon, alpha, gamma, n_planning)
    num_episodes = 300 # 智能体采样序列

    return_list = [] # 记录每一条序列的回报
    for i in range(10): # 显示10个进度条
        # 使用tqdm进度条
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)): # 每个进度条采样十分之一的序列
                # 序列初始化
                episode_return = 0
                state = env.reset()
                done = False
                # 采用真实数据、模拟数据更新
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done = env.step(action)
                    episode_return += reward # 不进行折扣因子处理
                    agent.update(state, action, reward, next_state) # 更新环境模型
                    state = next_state # 更新状态

                return_list.append(episode_return) # 记录序列奖励
                # 每10条序列打印下这10条序列的平均奖励
                if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    return return_list
