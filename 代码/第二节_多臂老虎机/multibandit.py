# 多臂老虎机
import numpy as np
import matplotlib.pyplot as plt

# 多臂老虎机
class BernoulliBandit:
    '''
    伯努利多臂老虎机
    '''
    def __init__(self, K):
        '''
        多臂老虎机的初始化
        
        输入:
        K:老虎机拉杆数
        '''
        # 各拉杆的中奖概率
        self.probs = np.random.uniform(size=K)
       	# 获奖概率最大的杆和对应概率
        self.best_idx = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_idx]
        # 杆数目
        self.K = K
    
    def step(self, k):
        '''
        玩家选择第k号杆子,依据拉杆获得奖励的概率返回是否获奖
        '''
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0

# 多臂老虎机求解
class Solver:
    '''
    多臂老虎机的求解
    '''
    def __init__(self, bandit):
        '''
        针对多臂老虎机bandit的求解
        '''
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K) # 初始化每根拉杆的尝试次数
        self.regret = 0 # 当前步的累积懊悔
        self.actions = [] # 记录每一步的行动
        self.regrets = [] # 记录每一步的累积懊悔

    def update_regret(self, k):
        '''
        通过当前步的动作，更新累积懊悔

        参数:
        k:当前步骤的动作(拉动第k个拉杆)
        '''
        # 此时的最优拉杆期望为bandit.best_prob
        # 懊悔定义为拉动当前杆和最优杆之间的奖励期望之差
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
        '''
        依据具体选择的策略，返回当前步选择的动作
        '''
        raise NotImplementedError
    
    def run(self, num_steps):
        '''
        进行num_steps次拉动杆
        '''
        for _ in range(num_steps):
            k = self.run_one_step() # 选择杆拉动
            self.counts[k] += 1 # 更新每根拉杆尝试记录
            self.actions.append(k) # 记录动作
            self.update_regret(k)

# 具体行动策略
# 1.ε-贪婪算法
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
        
# 直观显示累计函数
def plot_results(solvers, solver_names):
    '''
    绘制多种行动策略下的累积懊悔函数

    参数:
    solvers:策略组成的列表
    solver_names:策略列表名称
    '''
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()

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

# 随机生成多臂老虎机
np.random.seed(1) # 设定随机数种子
K = 10 # 10臂老虎机
bandit_10_arm = BernoulliBandit(K)
print("随机生成了一个%d臂伯努利老虎机" % K)
print("获奖概率最大的拉杆为%d号,其获奖概率为%.4f" % (bandit_10_arm.best_idx, bandit_10_arm.best_prob))

# 1.使用ε-贪婪算法
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
