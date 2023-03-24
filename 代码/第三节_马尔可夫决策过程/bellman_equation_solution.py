import numpy as np

# 定义随机数种子
np.random.seed(0)
# 定义状态转移概率矩阵P
P = [
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
    [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
]
P = np.array(P)

rewards = [-1, -2, -2, 10, 1, 0] # 定义各状态的奖励函数
gamma = 0.5 # 折扣因子

def compute(P, rewards, gamma, states_num):
    '''
    计算贝尔曼方程的解析解
    '''
    rewards = np.array(rewards).reshape((-1,1))
    value = np.dot(np.linalg.inv(np.eye(states_num,states_num) - gamma * P), rewards)

    return value

V = compute(P, rewards, gamma, 6)
print("MRP中每个状态价值分别为\n", V)
