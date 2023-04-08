import numpy as np
import matplotlib.pyplot as plt
from multibandit import BernoulliBandit, Solver, plot_results

class ThompsonSampling(Solver):
    '''
    汤普森采样算法(一种蒙特卡洛采样方法)
    '''
    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)
        self._a = np.ones(self.bandit.K) # 奖励为1的次数
        self._b = np.ones(self.bandit.K) # 奖励为0的次数

    def run_one_step(self):
        '''
        进行单步决策
        '''
        samples = np.random.beta(self._a, self._b) # 依据采样的奖励次数，构建beta分布，并采样后选择动作
        k = np.argmax(samples) # 选出采样奖励最大的拉杆
        r = self.bandit.step(k)

        # 更新beta分布的参数
        self._a[k] += r # 奖励为1的次数更新
        self._b[k] += (1-r) # 奖励为0的次数更新

        return k

if __name__ == "__main__":
    # 初始化10-臂老虎机
    np.random.seed(1) # 设定随机数种子
    K = 10 # 10臂老虎机
    bandit_10_arm = BernoulliBandit(K)

    # 使用汤普森采样算法
    np.random.seed(1)
    thompson_sampling_solver = ThompsonSampling(bandit_10_arm)
    thompson_sampling_solver.run(5000)
    print('汤普森采样算法的累积懊悔为：', thompson_sampling_solver.regret)
    plot_results([thompson_sampling_solver], ["ThompsonSampling"])
