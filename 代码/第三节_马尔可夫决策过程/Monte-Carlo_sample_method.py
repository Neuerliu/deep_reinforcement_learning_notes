import numpy as np

# 马尔可夫决策过程示例
# 定义状态和动作集合
S = ['s1', 's2', 's3', 's4', 's5']
A = ['保持s1', '前往s1', '前往s2', '前往s3', '前往s4', '前往s5', '概率前往']

# 状态转移函数
P = {
    "s1-保持s1-s1": 1.0,
    "s1-前往s2-s2": 1.0,
    "s2-前往s1-s1": 1.0,
    "s2-前往s3-s3": 1.0,
    "s3-前往s4-s4": 1.0,
    "s3-前往s5-s5": 1.0,
    "s4-前往s5-s5": 1.0,
    "s4-概率前往-s2": 0.2,
    "s4-概率前往-s3": 0.4,
    "s4-概率前往-s4": 0.4,
}

# 奖励函数
R = {
    "s1-保持s1": -1,
    "s1-前往s2": 0,
    "s2-前往s1": -1,
    "s2-前往s3": -2,
    "s3-前往s4": -2,
    "s3-前往s5": 0,
    "s4-前往s5": 10,
    "s4-概率前往": 1,
}

gamma = 0.5  # 折扣因子
MDP = (S, A, P, R, gamma) # 马尔可夫决策过程的集合

# 策略1,随机策略
Pi_1 = {
    "s1-保持s1": 0.5,
    "s1-前往s2": 0.5,
    "s2-前往s1": 0.5,
    "s2-前往s3": 0.5,
    "s3-前往s4": 0.5,
    "s3-前往s5": 0.5,
    "s4-前往s5": 0.5,
    "s4-概率前往": 0.5,
}

# 策略2
Pi_2 = {
    "s1-保持s1": 0.6,
    "s1-前往s2": 0.4,
    "s2-前往s1": 0.3,
    "s2-前往s3": 0.7,
    "s3-前往s4": 0.5,
    "s3-前往s5": 0.5,
    "s4-前往s5": 0.1,
    "s4-概率前往": 0.9,
}

def join(str1, str2):
    '''
    采用字符串拼接，拼接序列
    '''
    return str1 + '-' + str2

def sample(MDP, Pi, timestep_max, number):
    '''
    蒙特卡洛采样序列

    输入:
    MDP:马尔可夫决策过程的五元素(S, A, P, R, gamma)
    Pi:策略
    timestep_max:最大时间步长
    number:采样个数
    '''
    S, A, P, R, gamma = MDP # 解包马尔可夫决策过程
    episodes = [] # 采样列表

    # 采样
    for _ in range(number):
        # 采样一条从开始到结束的序列
        episode = []
        timestep = 0 # 时间步长初始化
        s = S[np.random.randint(len(S))] # 随机选择状态作为初始状态
        # 序列，定义终止状态为s5
        while s != "s5" and timestep <= timestep_max:
            timestep += 1
            # 采样动作
            rand, temp = np.random.rand(), 0 # 采样动作的随机概率，采样动作的实际概率初始化
            for a_opt in A:
                temp += Pi.get(join(s, a_opt), 0) # 从策略中采样一个状态动作对的概率，如果不存在该状态动作对，概率默认为0
                if temp > rand:
                    # 采用该状态动作对
                    a = a_opt
                    r = R.get(join(s, a), 0) # 奖励
                    break

            # 采样下一个状态
            rand, temp = np.random.rand(), 0 # 采样状态的随机概率，采样状态的实际概率初始化
            for s_opt in S:
                temp += P.get(join(join(s, a), s_opt), 0) # 采样状态-动作-状态的实际概率
                if temp > rand:
                    # 采用该状态-动作-状态
                    s_next = s_opt
                    break

            episode.append((s, a, r, s_next))
            s = s_next

        # 记录该采样序列
        episodes.append(episode)

    return episodes

# 采用策略1，采样5次，每个序列的最大时间步长为20
episodes = sample(MDP, Pi_1, 20, 5)
print("第一条序列\n", episodes[0])
print("第二条序列\n", episodes[1])
print("第三条序列\n", episodes[2])

# 对所有采样序列计算所有状态的价值
def MC(episodes, V, N, gamma):
    ''''
    依据多条采样序列，计算每个状态的价值

    输入:
    episodes:多条采样序列
    V:状态价值列表
    N:状态被访问的次数
    gamma:折扣因子
    '''
    for episode in episodes:
        G = 0 # 当前序列的回报
        # 序列从后往前计算
        for i in range(len(episode)-1, -1, -1):
            (s, a, r, s_next) = episode[i] # 当前时间步解包
            G = r + gamma * G # 奖励折扣
            N[s] = N[s] + 1 # 状态s访问次数更新
            V[s] = V[s] + (G - V[s]) / N[s] # 状态价值更新

timestep_max = 20
# 采样1000次,可以自行修改
episodes = sample(MDP, Pi_1, timestep_max, 1000)
V = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}
N = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}
MC(episodes, V, N, gamma)
print("使用蒙特卡洛方法计算MDP的状态价值为\n", V)
