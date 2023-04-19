import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # 显示循环进度条的库

from sarsa_cliff_walking_env import CliffWalkingEnv

# 定义Sarsa类
class Sarsa:
    '''
    Sarsa算法
    '''
    # Sarsa算法的初始化
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action]) # 维护一个动作价值函数的表格
        self.n_action = n_action # 动作个数
        self.alpha = alpha # 学习率
        self.gamma = gamma # 折扣因子
        self.epsilon = epsilon # epsilon-贪婪中的参数

    # 依据当前状态选择动作
    def take_action(self, state):
        # 采用epsilon-贪婪策略
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])

        return action
    
    # 状态state下的最优动作
    def best_action(self, state):
        Q_max = np.max(self.Q_table[state]) # 最大动作价值
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:
                a[i] = 1

        return a
    
    # 更新动作价值函数
    def update(self, s0, a0, r, s1, a1):
        td_error = r + self.gamma * self.Q_table[s1, a1] - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error

# 打印Sarsa的策略
def print_agent(agent, env, action_meaning, disaster=[], end=[]):
    '''
    打印时序差分算法的策略
    '''
    for i in range(env.nrow):
        for j in range(env.ncol):
            if (i * env.ncol + j) in disaster:
                # 到达悬崖位置
                print('****', end=' ')
            elif (i * env.ncol + j) in end:
                # 到达终点
                print('EEEE', end=' ')
            else:
                # 其他位置
                a = agent.best_action(i * env.ncol + j) # 当前状态的最优策略
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print("")

if __name__ == '__main__':
    ncol = 12
    nrow = 4
    env = CliffWalkingEnv(ncol, nrow) # 没有明确概率转移矩阵和奖励函数下的悬崖漫步环境
    np.random.seed(0) # 设置随机数种子
    epsilon = 0.1 # epsilon-贪婪的参数
    alpha = 0.1 # 学习率
    gamma = 0.9 # 折扣因子
    agent = Sarsa(ncol, nrow, epsilon, alpha, gamma)
    num_episodes = 500  # 智能体在环境中运行的序列的数量

    return_list = []
    for i in range(10):
        # tqdm进度条
        # 显示10个进度条
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            # 每个进度条50个序列
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0 # 当前序列的回报
                state = env.reset() # 重置状态
                action = agent.take_action(state)
                done = False

                # 采样到序列结束
                while not done:
                    # 进行一次SARSA
                    next_state, reward, done = env.step(action) # 环境给出奖励和下一阶段的状态
                    next_action = agent.take_action(next_state)
                    episode_return += reward # 这里的回报计算不进行折扣因子衰减
                    agent.update(state, action, reward, next_state, next_action) # 更新动作价值函数
                    # 更新state和action
                    state = next_state
                    action = next_action

                # 记录序列的回报
                return_list.append(episode_return)

                # 每10条序列打印下平均回报
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return': '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

# 打印回报变化过程
# 此时的回报并不要折扣因子衰减
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Sarsa on {}'.format('Cliff Walking'))
plt.show()

action_meaning = ['^', 'v', '<', '>']
print('Sarsa算法最终收敛得到的策略为：')
print_agent(agent, env, action_meaning, list(range(37, 47)), [47])
