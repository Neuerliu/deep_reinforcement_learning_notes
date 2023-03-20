import numpy as np
import matplotlib.pyplot as plt

from multibandit import BernoulliBandit, Solver, plot_results

# 具体行动策略
# ε-贪婪算法
class EpsilonGreedy(Solver):
    '''
    epsilon-贪婪算法
    '''
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        # 超类初始化
        super(EpsilonGreedy, self).__init__(bandit)
        # 探索概率
        self.epsilon = epsilon
        # 初始化拉杆的期望奖励估值
        self.estimates = np.array([init_prob] * self.bandit.K)

    def run_one_step(self):
        '''
        每一步行动
        '''
        # 拉动拉杆
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.K) # 随机拉动一根拉杆
        else:
            k = np.argmax(self.estimates) # 当前期望奖励最大的拉杆

        r = self.bandit.step(k) # 得到当前奖励
        # 更新期望奖励
        self.estimates[k] += 1/(self.counts[k]+1) * (r - self.estimates[k])

        return k

# 采用随时间衰减的epsilon-贪婪算法
class DecayingEpsilonGreedy(Solver):
    '''
    随时间衰减的epsilon-贪婪算法
    '''
    def __init__(self, bandit, init_prob=1.0):
        # 超类初始化
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        # 当前进行实验次数
        self.total_count = 0
        # 初始化拉杆的期望奖励估值
        self.estimates = np.array([init_prob] * self.bandit.K)

    def run_one_step(self):
        '''
        每一步行动
        '''
        self.total_count += 1
        # 拉动拉杆
        if np.random.random() < 1/self.total_count:
            k = np.random.randint(0, self.bandit.K) # 随机拉动一根拉杆
        else:
            k = np.argmax(self.estimates) # 当前期望奖励最大的拉杆

        r = self.bandit.step(k) # 得到当前奖励
        # 更新期望奖励
        self.estimates[k] += 1/(self.counts[k]+1) * (r - self.estimates[k])

        return k
    
# 初始化10-臂老虎机
np.random.seed(1) # 设定随机数种子
K = 10 # 10臂老虎机
bandit_10_arm = BernoulliBandit(K)

# 使用ε-贪婪算法
np.random.seed(1)
epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.01)
epsilon_greedy_solver.run(5000) # 进行5000步的尝试
print('epsilon-贪婪算法的累积懊悔为：', epsilon_greedy_solver.regret)
plot_results([epsilon_greedy_solver], ['EpsilonGreedy'])

# 比较不同epsilon值下的累积懊悔变化
np.random.seed(0)
epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
epsilon_greedy_solver_list = [
    EpsilonGreedy(bandit_10_arm, epsilon=epsilon) for epsilon in epsilons
]
epsilon_greedy_solver_names = [
    'epsilon={}'.format(epsilon) for epsilon in epsilons 
]
for solver in epsilon_greedy_solver_list:
    solver.run(5000)

plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)

# 使用随时间衰减的epsilon-贪婪算法
np.random.seed(1)
decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
decaying_epsilon_greedy_solver.run(5000) # 进行5000步的尝试
print('epsilon衰减的贪婪算法的累积懊悔为：', decaying_epsilon_greedy_solver.regret)
plot_results([decaying_epsilon_greedy_solver], ['DecayingEpsilonGreedy'])
