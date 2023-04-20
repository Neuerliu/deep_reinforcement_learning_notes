import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sarsa_cliff_walking_env import CliffWalkingEnv

class nstep_sarsa:
    '''
    多步Sarsa算法

    输入:
    :param n: n步
    :param ncol: 列数
    :param nrow: 行数
    :param epsilon: epsilon-greedy中的参数
    :param alpha: 学习率
    :param gamma: 折扣因子
    :param n_action: 动作个数
    '''
    def __init__(self, n, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action]) # 动作价值函数表格
        self.n_action = n_action # 动作个数
        self.alpha = alpha # 学习率
        self.gamma = gamma # 折扣因子
        self.epsilon = epsilon # epsilon-greedy中的参数
        self.n = n # 采用n步Sarsa算法

        # 由于需要n步Sarsa，需要记录n步过程，用于后续更新动作价值函数
        self.state_list = [] # 保存之前的状态
        self.action_list = [] # 保存之前的动作
        self.reward_list = [] # 保存之前的奖励

    # 采用epsilon-greedy策略
    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action) # 采用随机策略
        else:
            action = np.argmax(self.Q_table[state]) # 采用贪婪策略

        return action
    
    # 打印策略，即展示状态state下的最优行动
    def best_action(self, state):
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)] # 记录state下的行动
        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:
                a[i] = 1

        return a
    
    # 更新动作价值函数，即更新Q_table
    def update(self, s0, a0, r, s1, a1, done):
        # 加上状态s0, a0
        self.state_list.append(s0)
        self.action_list.append(a0)
        self.reward_list.append(r)

        # 如果已有n步的数据
        if len(self.state_list) == self.n:
            G = self.Q_table[s1, a1] # 得到Q(s_{t+n}, a_{t+n})
            # 从n步往前推
            for i in reversed(range(self.n)):
                G = self.gamma * G + self.reward_list[i] # 向前计算

                # 如果序列最后达到终止，即使最后几步的后面没有n步，也可以采用多步Sarsa更新
                if done and i > 0:
                    s = self.state_list[i]
                    a = self.action_list[i]
                    self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a]) # 多步Sarsa的更新公式

            # 更新列表中第0个状态并删除
            s = self.state_list.pop(0)
            a = self.action_list.pop(0)
            self.reward_list.pop(0)
            # 多步Sarsa的更新
            self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a])

        # 如果是终止状态，重新采样序列
        if done:
            self.state_list = []
            self.action_list = []
            self.reward_list = []

if __name__ == '__main__':
    ncol = 12
    nrow = 4
    np.random.seed(0)
    n_step = 5
    alpha = 0.1
    epsilon = 0.1
    gamma = 0.9
    agent = nstep_sarsa(n_step, ncol, nrow, epsilon, alpha, gamma)
    env = CliffWalkingEnv(ncol, nrow)
    num_episodes = 500 # 智能体采样序列数目

    return_list = []
    for i in range(10):
        # tqdm进度条
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):  # 每个进度条的序列数
                episode_return = 0 # 当前序列的回报
                state = env.reset()
                action = agent.take_action(state)
                done = False

                # 采样一条序列
                while not done:
                    next_state, reward, done = env.step(action)
                    next_action = agent.take_action(next_state)
                    episode_return += reward  # 这里回报的计算不进行折扣因子衰减
                    agent.update(state, action, reward, next_state, next_action, done)
                    state = next_state
                    action = next_action
                return_list.append(episode_return)
                
                if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('5-step Sarsa on {}'.format('Cliff Walking'))
    plt.show()
