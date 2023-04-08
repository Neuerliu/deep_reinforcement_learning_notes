import numpy as np

from bellman_equation_solution import compute

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

# 结合策略和状态转移函数，将MDP下的状态转移函数，转化为MRP下状态转移函数
def convert_P_mdp2mrp(P_mdp, Pi, S, A):
    '''
    将给定策略下的MDP的状态转移函数转化为MRP的状态转移函数

    输入:
    P_mdp:MDP下的状态转移函数
    Pi:给定策略
    S:状态列表
    A:动作列表
    '''
    state_len = len(S)
    P_mrp = np.zeros((state_len, state_len)) # MRP下的状态转移函数初始化
    for state_start in S:
        state_start_idx = S.index(state_start) # 起始状态索引
        for action in A:
            cur_state_action_pair = state_start + '-' + action# 当前的状态动作对

            # 找到策略中的状态动作对
            if cur_state_action_pair in Pi.keys():
                possibility = Pi[cur_state_action_pair] # 指定动作的概率
                state_end_dict = {k.split(cur_state_action_pair)[1][1:]:v for k, v in P_mdp.items() if cur_state_action_pair in k}

                for state_end, reward in state_end_dict.items():
                    state_end_idx = S.index(state_end) # 终止状态索引
                    P_mrp[state_start_idx, state_end_idx] += possibility * reward

    return P_mrp

def convert_R_mdp2mrp(R_mdp, Pi, S, A):
    '''
    将给定策略下的MDP的状态转移函数转化为MRP的状态转移函数

    输入:
    P_mdp:MDP下的状态转移函数
    Pi:给定策略
    S:状态列表
    A:动作列表
    '''
    state_len = len(S)
    R_mrp = np.zeros(state_len) # MRP下的状态转移函数初始化
    for state_start in S:
        state_start_idx = S.index(state_start) # 起始状态索引
        for action in A:
            cur_state_action_pair = state_start + '-' + action# 当前的状态动作对

            # 找到策略中的状态动作对
            if cur_state_action_pair in Pi.keys():
                cur_state_action_pair_idx = [pi_key for pi_key in Pi.keys() if cur_state_action_pair in pi_key][0]
                possibility = Pi[cur_state_action_pair] # 指定动作的概率
                R_mrp[state_start_idx] += possibility * R_mdp[cur_state_action_pair_idx]

    return R_mrp

# print(convert_P_mdp2mrp(P, Pi_1, S, A))
# print(convert_R_mdp2mrp(R, Pi_1, S, A))
if __name__ == "__main__":
    # 求解策略1下的状态转移函数，以此求解状态价值函数
    P_from_mdp_to_mrp = convert_P_mdp2mrp(P, Pi_1, S, A)
    R_from_mdp_to_mrp = convert_R_mdp2mrp(R, Pi_1, S, A)
    V = compute(P_from_mdp_to_mrp, R_from_mdp_to_mrp, gamma, 5)
    print("策略1 MDP中每个状态价值分别为\n", V)

    # 求解策略2下的状态转移函数，以此求解状态价值函数
    P_from_mdp_to_mrp = convert_P_mdp2mrp(P, Pi_2, S, A)
    R_from_mdp_to_mrp = convert_R_mdp2mrp(R, Pi_2, S, A)
    V = compute(P_from_mdp_to_mrp, R_from_mdp_to_mrp, gamma, 5)
    print("策略2 MDP中每个状态价值分别为\n", V)
