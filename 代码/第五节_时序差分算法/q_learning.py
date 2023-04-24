import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sarsa_utils import CliffWalkingEnv, print_agent

class QLearning:
    '''
    Q-Learning算法
    '''
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action]) # 初始化最优动作价值函数
        self.n_action = n_action # 动作个数
        self.alpha = alpha # 学习率
        self.gamma = gamma # 折扣因子
        self.epsilon = epsilon # epsilon-greedy策略中的参数

    # 动作选择 -> 采样状态动作对(s,a)
    def take_action(self, state):
        # 采用epsilon-greedy策略
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        
        return action

    # 最优动作 -> 便于之后打印动作
    def best_action(self, state):
        Q_max = np.max(self.Q_table[state])
        a = [0 for i in range(self.n_action)] # 初始化动作
        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:
                a[i] = 1

        return a
    
    def update(self, s0, a0, r, s1):
        # 更新最优动作价值函数时采用的策略是纯贪婪策略
        td_error = r + self.gamma * self.Q_table[s1].max() - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error

if __name__ == '__main__':
    ncol = 12
    nrow = 4
    np.random.seed(0)
    epsilon = 0.1
    alpha = 0.1
    gamma = 0.9
    agent = QLearning(ncol, nrow, epsilon, alpha, gamma)
    env = CliffWalkingEnv(ncol, nrow)
    num_episodes = 500  # 智能体在环境中运行的序列的数量

    return_list = []  # 记录每一条序列的回报
    for i in range(10):  # 显示10个进度条
        # tqdm的进度条功能
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):  # 每个进度条的序列数
                episode_return = 0
                state = env.reset()
                done = False
                while not done: # 采样一个序列
                    action = agent.take_action(state) # 采用epsilon-greedy策略采样当前状态下动作
                    next_state, reward, done = env.step(action) # 环境给出下个状态和奖励
                    episode_return += reward  # 这里回报的计算不进行折扣因子衰减
                    agent.update(state, action, reward, next_state) # 目标策略为纯贪婪策略(目标策略就是更新动作价值函数的策略)
                    state = next_state
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
    plt.title('Q-learning on {}'.format('Cliff Walking'))
    plt.show()

    action_meaning = ['^', 'v', '<', '>']
    print('Q-learning算法最终收敛得到的策略为：')
    print_agent(agent, env, action_meaning, list(range(37, 47)), [47])
