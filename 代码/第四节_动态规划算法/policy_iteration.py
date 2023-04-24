import copy
from dp_utils import CliffWalkingEnv, print_agent

# 基于悬崖漫步环境下的策略梯度法
# 策略迭代
class PolicyIteration:
    '''
    策略迭代算法
    '''
    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [0] * self.env.ncol * self.env.nrow # 状态价值函数初始化
        self.pi = [
            [0.25, 0.25, 0.25, 0.25] for i in range(self.env.ncol * self.env.nrow)
        ] # 策略的初始化
        self.theta = theta # 策略评估收敛阈值
        self.gamma = gamma # 折扣因子

    # 策略评估
    def policy_evaluation(self):
        cnt = 1 # 计数器
        while 1:
            max_diff = 0
            new_v = [0] * self.env.ncol * self.env.nrow # 新的状态价值函数初始化
            # 遍历状态，更新状态价值函数
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = [] # 当前状态s下所有Q(s,a)的值
                # 遍历动作
                for a in range(4):
                    qsa = 0
                    # 执行当前动作
                    # 遍历此时的状态转移函数
                    for res in self.env.P[s][a]:
                        # [(p, next_state, reward, done)] 
                        # 对应(转移概率, 下一个状态, 奖励, 是否处于终止状态)
                        p, next_state, r, done = res
                        # 悬崖漫步环境比较特殊，当前步的奖励和下一个状态有关
                        qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))

                    # 状态s下的所有的动作价值在策略pi下的期望
                    qsa_list.append(self.pi[s][a] * qsa)
                # 更新价值函数
                new_v[s] = sum(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s])) # 状态价值函数更新前后的差别
            
            self.v = new_v # 更新状态价值函数
            if max_diff < self.theta:
                break # 满足收敛条件，则退出策略评估
            cnt += 1

        print("策略评估进行%d轮后完成" % cnt)

    # 策略改进
    def policy_improvement(self):
        # 遍历状态
        for s in range(self.env.nrow * self.env.ncol):
            qsa_list = []
            # 遍历动作
            for a in range(4):
                qsa = 0
                # 遍历(s,a)下状态转移函数
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                qsa_list.append(qsa)

            # 选择动作价值最大的动作
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq)  # 计算有几个动作得到了最大的Q值
            # 让这些动作均分概率
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]
        print("策略提升完成")
        return self.pi
    
    # 策略迭代
    def policy_iteration(self):
        while 1:
            self.policy_evaluation() # 策略评估
            old_pi = copy.deepcopy(self.pi) # 深拷贝策略
            new_pi = self.policy_improvement()
            if old_pi == new_pi:
                break # 停止策略迭代

if __name__ == '__main__':
    env = CliffWalkingEnv()
    action_meaning = ['^', 'v', '<', '>'] # 动作含义
    theta = 0.001
    gamma = 0.9
    # 采用策略迭代的智能体
    agent = PolicyIteration(env, theta, gamma)
    agent.policy_iteration()
    print_agent(agent, action_meaning, list(range(37, 47)), [47])
