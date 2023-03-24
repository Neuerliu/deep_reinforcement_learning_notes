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

# 计算给定序列上，从某个索引开始到序列结束的回报
def compute_return(start_index, chain, gamma):
    '''
    计算回报，采用回溯计算的方式
    '''
    G = 0 # 总回报
    for i in reversed(range(start_index, len(chain))):
        G = G * gamma + rewards[chain[i]-1]

    return G

# 一个状态序列,s1-s2-s3-s6
chain = [1, 2, 3, 6]
start_index = 0
G = compute_return(start_index, chain, gamma)
print("根据本序列计算得到回报为：%s。" % G)
