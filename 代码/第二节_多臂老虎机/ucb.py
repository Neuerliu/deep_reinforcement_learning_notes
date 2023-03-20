import numpy as np
import matplotlib.pyplot as plt

from multibandit import BernoulliBandit, Solver, plot_results

# 采用上置信界算法
class UCB(Solver):
    '''
    上置信界算法

    参数:
    bandit:多臂老虎机实例
    coef:不确定性度量在目标函数中的参数
    init_prob:每个杆的期望奖励概率初始化
    '''
    # 初始化
    def __init__(self, bandit, coef, init_prob=1.0):
        super(UCB, self).__init__(bandit)
        self.total_count = 0 # 记录当前时间步
        self.estimates = np.array([init_prob] * self.bandit.K) # 对每个杆的期望奖励估计
        self.coef = coef # 不确定性度量的参数

    # 选择该步的动作
    # 使用期望奖励上界最大的杆
    def run_one_step(self):
        self.total_count += 1
        exceed_prob = 1/self.total_count # 超过期望奖励上界的概率(与时间步成反比)
        ucb = self.estimates + self.coef * np.sqrt(
            - np.log(exceed_prob) / (2 * (self.counts + 1))
        )
        # 选取期望奖励上界最大的杆子
        k = np.argmax(ucb)
        r = self.bandit.step(k)
        self.estimates[k] += 1 / (self.counts[k]+1) * (r - self.estimates[k]) # 期望奖励更新公式

        return k

# 初始化10-臂老虎机
np.random.seed(1) # 设定随机数种子
K = 10 # 10臂老虎机
bandit_10_arm = BernoulliBandit(K)

np.random.seed(1)
coef = 1 # 控制不确定性比重的系数
UCB_solver = UCB(bandit_10_arm, coef)
UCB_solver.run(5000)
print('上置信界算法的累积懊悔为：', UCB_solver.regret)
plot_results([UCB_solver], ["UCB"])
