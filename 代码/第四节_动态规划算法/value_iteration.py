import numpy as np
from cliff_walking_env import CliffWalkingEnv, print_agent

class ValueIteration:
    '''
    价值迭代算法

    输入:
    :param env: 环境
    :param theta: 策略评估收敛阈值
    :param gamma: 折扣因子
    '''
    def __init__(self, env, theta, gamma):
        self.env = env # 悬崖漫步环境
        self.v = [0] * self.env.ncol * self.env.nrow # 初始化状态价值函数
        self.theta = theta # 价值函数收敛的阈值
        self.gamma = gamma # 折扣因子
        # 价值迭代结束后得到的策略
        self.pi = [None for i in range(self.env.ncol * self.env.nrow)]

    # 价值迭代
    def value_iteration(self):
        cnt = 0
        while 1:
            max_diff = 0 # 记录价值迭代的次数
            new_v = [0] * self.env.ncol * self.env.nrow # 贝尔曼最优性方程更新最优价值函数
            # 遍历状态
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = [] # 计算状态s下Q(s, a)的值
                # 遍历动作
                for a in range(4):
                    qsa = 0
                    # 遍历状态转移函数
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res
                        # 当前Q(s,a)的值
                        qsa += p * (r + self.gamma * self.v[next_state] * (1-done))
                    qsa_list.append(qsa)

                new_v[s] = max(qsa_list) # 贝尔曼最优性方程，更新最优价值函数
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            # 如果价值函数收敛
            if max_diff < self.theta:
                break
            cnt += 1
        print("价值迭代一共进行%d轮" % cnt)
        self.get_policy()

    # 获取策略
    def get_policy(self):
        # 计算给定状态下的动作价值函数
        for s in range(self.env.nrow * self.env.ncol):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_state] * (1-done))
                qsa_list.append(qsa)

            # 依据当前状态下最大动作状态价值选择动作，作为策略
            maxq = max(qsa_list) # 最大价值
            cntq = qsa_list.count(maxq) # 最大价值的动作个数
            # 具有最大动作价值的动作均分概率，作为策略
            self.pi[s] = [1/cntq if q == maxq else 0 for q in qsa_list]

# 采用悬崖漫步环境
env = CliffWalkingEnv()
action_meaning = ['^', 'v', '<', '>'] # 动作含义
theta = 0.001
gamma = 0.9
agent = ValueIteration(env, theta, gamma) # 价值迭代
agent.value_iteration()
print_agent(agent, action_meaning, list(range(37, 47)), [47])
