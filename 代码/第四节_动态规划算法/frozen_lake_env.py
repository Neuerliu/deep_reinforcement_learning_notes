import gym
from policy_iteration import PolicyIteration
from value_iteration import ValueIteration
from dp_utils import print_agent

# 使用gym环境
env = gym.make('FrozenLake-v1', render_mode='rgb_array') # 还可以使用'human'
env = env.unwrapped # 解封装以访问状态转移矩阵
env.reset() # 重置环境并返回初始状态

env.render() # 环境渲染，通常是弹窗显示或者打印出可视化环境

holes = set() # 冰洞的集合初始化
ends = set() # 终点的集合初始化
for s in env.P: # 状态空间
    for a in env.P[s]: # 状态s下的动作空间
        # 遍历(s,a)下的概率转移矩阵
        for s_ in env.P[s][a]:
            # s_的形式为(p, next_state, r, is_hole)
            # 对应(转移概率, 下一个状态, 当前步奖励, 是否为终止状态)
            if s_[2] == 1.0: # 获得奖励为1 -> 终点
                ends.add(s_[1])
            if s_[3] == True: # 为终止状态
                holes.add(s_[1])

holes = holes - ends # 冰洞需要排除终点
print('冰洞的索引为:', holes)
print('目标的索引为:', ends)

# 终点左边一格的状态下各种可能动作
for a in env.P[14]:
    print(env.P[14][a])

# 尝试下策略迭代算法
print('-----------------------------------------')
print('策略迭代:')
action_meaning = ['<', 'v', '>', '^'] # 动作含义
theta = 1e-5 # 策略迭代阈值
gamma = 0.9 # 折扣因子
agent = PolicyIteration(env, theta, gamma)
agent.policy_iteration() # 策略迭代
print_agent(agent, action_meaning, list(holes), list(ends))

# 价值迭代
print('-----------------------------------------')
print('价值迭代:')
agent = ValueIteration(env, theta, gamma)
agent.value_iteration()
print_agent(agent, action_meaning, list(holes), list(ends))
