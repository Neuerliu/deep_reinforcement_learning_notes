import numpy as np

from Monte_Carlo_sample_method import sample

def occupancy(episodes, s, a, timestep_max, gamma):
    '''
    计算状态动作对(s, a)的占用度量

    输入:
    episodes:采样的序列构成的列表
    s:状态
    a:动作
    timestep_max:采样序列的最大时间步长
    gamma:折扣因子
    '''
    rho = 0 # 占用度量初始化为0
    total_times = np.zeros(timestep_max) # 记录每个时间步t在采样序列中出现的次数
    occur_times = np.zeros(timestep_max) # 记录(s_t, a_t) = (s, a)的次数
    # 遍历采样序列表
    for episode in episodes:
        # 遍历当前序列各阶段
        for i in range(len(episode)):
            (s_opt, a_opt, r, s_next) = episode[i] # 访问序列当前状态、动作、奖励
            total_times[i] += 1 # 时间步i在当前采样序列中出现一次
            if s == s_opt and a == a_opt:
                occur_times[i] += 1

    # 采用倒序方式更新rho，更加方便
    for i in reversed(range(timestep_max)):
        if total_times[i]:
            rho += gamma ** i * occur_times[i] / total_times[i]

    return (1-gamma) * rho


if __name__ == "__main__":
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

    # 采样
    timestep_max = 1000
    episodes_1 = sample(MDP, Pi_1, timestep_max, 1000) # 采样1000个
    episodes_2 = sample(MDP, Pi_2, timestep_max, 1000) # 采样1000个

    # 计算占用度量
    rho_1 = occupancy(episodes_1, 's4', '概率前往', timestep_max, gamma)
    rho_2 = occupancy(episodes_2, 's4', '概率前往', timestep_max, gamma)
    print(rho_1, rho_2)
