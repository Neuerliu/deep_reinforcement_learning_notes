# Dyna-Q算法

## 6.1 简介

在强化学习中，“模型”通常指与智能体交互的环境模型，即对环境的**状态转移概率**和**奖励函数**进行建模。根据是否具有环境模型，强化学习算法分为两种：

- **基于模型的强化学习**(model-based reinforcement learning)：模型可以是事先知道的，也可以是根据智能体与环境交互采样到的数据学习得到的，然后用这个模型帮助策略提升或者价值估计，例如动态规划算法，即策略迭代和价值迭代。
- **无模型的强化学习**(model-free reinforcement learning)：根据智能体与环境交互采样到的数据直接进行策略提升或者价值估计，例如时序差分算法，即`Sarsa`和Q-learning算法。

Dyna-Q算法也是一种基于模型的强化学习算法，不过它的环境模型是通过采样数据估计得到的。

强化学习算法有两个重要的评价指标：算法收敛后的策略在初始状态下的期望回报、样本复杂度(算法达到收敛结果需要在真实环境中采样的样本数量)。基于模型的强化学习算法由于具有一个环境模型，智能体可以额外和环境模型进行交互，对真实环境中样本的需求量往往就会减少，因此通常会**比无模型的强化学习算法具有更低的样本复杂度**。但是，环境模型可能并不准确，不能完全代替真实环境，因此基于模型的强化学习算法收敛后其策略的期望回报可能**不如**无模型的强化学习算法。

## 6.2 Dyna-Q算法

Dyna-Q使用一种叫做Q-planning的方法来基于模型生成一些**模拟数据**，然后用**模拟数据和真实数据一起改进策略**。Q-planning每次选取一个曾经访问过的状态 $s$，采取一个曾经在该状态下执行过的动作 $a$，通过模型得到转移后的状态 $s^{\prime}$ 以及奖励 $r$，并根据这个模拟数据 $(s, a, r, s^{\prime})$，用Q-learning的更新方式来更新动作价值函数。

Dyna-Q算法的大致流程如下所示：

![img](https://hrl.boyuai.com/static/480.25b67b37.png)

以下展示Dyna-Q算法的伪代码：

初始化 $Q(s, a)$，初始化模型 $M(s, a)$

$for$ 序列$e=1 \rightarrow E \space do$ :

​	得到初始状态 $s$

​	$for \space t=1 \rightarrow T \space do$ :

​		用 $\epsilon-greedy$ 策略根据 $Q$ 选择当前状态 $s$ 下的动作 $a$ 

​		得到环境反馈的 $r, s^{\prime}$

​		// 采用真实样本更新动作价值函数

​		$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \mathop{max}_{a^{\prime}}{Q(s^{\prime}, a^{\prime})} - Q(s, a)]$

​		// 采用真实样本更新拟合的环境模型

​		$M(s, a) \leftarrow r, s^{\prime}$

​		$for$ 次数 $n=1 \rightarrow N \space do$ :

​			随机选择一个曾经访问过的状态 $s_m$

​			采取一个曾经在状态 $s_m$ 执行过的动作 $a_m$

​			// 采用模型生成的模拟数据更新动作价值函数

​			$r_m, s^{\prime}_m \leftarrow M(s_m, a_m)$

​			$Q(s_m, a_m) \leftarrow Q(s_m, a_m) + \alpha [r_m + \gamma \mathop{max}_{a^{\prime}}{Q(s^{\prime}_m, a^{\prime})} - Q(s_m, a_m)]$

​		$end \space for$

​		$s \leftarrow s^{\prime}$

​	$end \space for$

$end \space for$	

在每次与环境进行交互执行一次Q-learning之后，Dyna-Q算法会做 $n$ 次Q-planning。其中Q-planning的次数 $n$ 是一个事先可以选择的超参数，**当其为 0 时就是普通的Q-learning**。值得注意的是，上述Dyna-Q算法是执行在一个离散并且确定的环境中，所以当看到一条经验数据 $(s, a, r, s^{\prime})$ 时，可以直接对模型做出更新，即 $M(s, a) \leftarrow r, s^{\prime}$。

## 6.3 Dyna-Q代码实践

我们在悬崖漫步环境中实现Dyna-Q算法，首先实现悬崖漫步的环境代码：

```python
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import time

class CliffWalkingEnv:
    '''
    定义Dyna-Q的悬崖漫步环境
    '''
    def __init__(self, ncol, nrow):
        # 行列初始化
        self.nrow = nrow
        self.ncol = ncol
        self.x = 0 # 当前智能体的横坐标
        self.y = self.nrow - 1 # 当前智能体的纵坐标

    # 环境和智能体交互函数
    def step(self, action):
        # 4种动作, change[0]:上, change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        # 定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0])) # 不能超过横坐标限制
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1])) # 不能超过纵坐标限制
        next_state = self.y * self.ncol + self.x # 下一个状态
        reward = -1
        done = False
        if self.y == self.nrow - 1 and self.x > 0:  # 下一个位置在悬崖或者目标
            done = True
            # 下一个状态在悬崖
            if self.x != self.ncol - 1:
                reward = -100
        
        return next_state, reward, done

    # 重置函数
    def reset(self):  # 回归初始状态,起点在左上角
        self.x = 0
        self.y = self.nrow - 1

        return self.y * self.ncol + self.x
```

接下来定义智能体和不同`n_planning`下的训练函数：

```python

# 定义dyna-Q智能体
class DynaQ:
    '''
    Dyna-Q算法

    输入:
    :param ncol: 列数
    :param nrow: 行数
    :param epsilon: epsilon-greedy中的参数
    :param alpha: 学习率
    :param gamma: 折扣因子
    :param n_planning: 使用模型模拟数据更新Q函数的次数
    :param n_action: 动作个数
    '''
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_planning, n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action]) # 初始化动作价值函数
        self.n_action = n_action # 动作个数
        self.alpha = alpha # 学习率
        self.gamma = gamma # 折扣因子
        self.epsilon = epsilon # epsilon-greedy中的参数

        self.n_planning = n_planning # 使用模型模拟数据更新Q函数的次数，对应一次Q-learning
        self.model = dict() # 初始化模型

    # 采用epsilon-greedy策略选择动作
    def take_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action
    
    # 使用采样得到的真实数据更新Q函数
    def q_learning(self, s0, a0, r, s1):
        td_error = r + self.gamma * np.max(self.Q_table[s1]) - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error

    # 完整的更新动作价值函数
    def update(self, s0, a0, r, s1):
        self.q_learning(s0, a0, r, s1)
        self.model[(s0, a0)] = r, s1 # 利用采样数据更新环境模型
        for _ in range(self.n_planning):
            # 从环境的字典中随机选择曾经遇到的状态-动作对
            (s, a), (r, s_) = random.choice(list(self.model.items()))
            self.q_learning(s, a, r, s_) # 利用模型模拟数据更新Q函数

# 定义dyna-Q的训练函数
def DynaQ_CliffWalking(n_planning):
    ncol = 12
    nrow = 4
    env = CliffWalkingEnv(ncol, nrow)
    epsilon = 0.01
    alpha = 0.1
    gamma = 0.9
    agent = DynaQ(ncol, nrow, epsilon, alpha, gamma, n_planning)
    num_episodes = 300 # 智能体采样序列

    return_list = [] # 记录每一条序列的回报
    for i in range(10): # 显示10个进度条
        # 使用tqdm进度条
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)): # 每个进度条采样十分之一的序列
                # 序列初始化
                episode_return = 0
                state = env.reset()
                done = False
                # 采用真实数据、模拟数据更新
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done = env.step(action)
                    episode_return += reward # 不进行折扣因子处理
                    agent.update(state, action, reward, next_state) # 更新环境模型
                    state = next_state # 更新状态

                return_list.append(episode_return) # 记录序列奖励
                # 每10条序列打印下这10条序列的平均奖励
                if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    return return_list
```

结果展示如下：

```python
if __name__ == '__main__':
    # 展示结果
    np.random.seed(0)
    random.seed(0)
    n_planning_list = [0, 2, 20] # 不同n_planning
    for n_planning in n_planning_list:
        print('Q-planning步数为：%d' % n_planning)
        time.sleep(0.5)
        return_list = DynaQ_CliffWalking(n_planning)
        episodes_list = list(range(len(return_list)))
        plt.plot(episodes_list, return_list, label=str(n_planning) + ' planning steps')

    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Dyna-Q on {}'.format('Cliff Walking'))
    plt.show()
```

![](D:\动手学强化学习\deep_reinforcement_learning_notes\img\第六节_Dyna_Q_算法\Dyna_Q_cliffwalking.png)

从上述结果中我们可以看出：**随着Q-planning步数的增多，Dyna-Q算法的收敛速度也随之变快**。在上述悬崖漫步环境中，状态的转移是完全确定性的，构建的环境模型的精度是最高的，所以可以通过增加Q-planning步数来直接降低算法的样本复杂度。
