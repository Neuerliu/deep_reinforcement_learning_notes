# `DQN`算法

## 7.1 简介

在前面的算法中，我们采用的环境(悬崖漫步环境、冰湖环境)均具有离散的状态动作空间，可以用矩阵存储动作价值函数，但是当状态或者动作数量非常大的时候，这种做法就不适用了。例如，当**状态是一张RGB图像**时，假设图像大小为 $210 \times 160 \times 3$，此时一共有 $256^{210 \times 160 \times 3}$ 种状态，在计算机中存储这个数量级的值表格是不现实的。此外，**当状态或者动作连续**的时候，就有无限个状态动作对，我们更加无法使用这种表格形式来记录各个状态动作对的值。

对于这种情况，我们需要用**函数拟合**的方法来估计值，即将这个复杂的值表格视作数据，使用一个参数化的函数来拟合这些数据。很显然，这种函数拟合的方法存在一定的精度损失，因此被称为**近似方法**。`DQN`算法就是一种近似方法，可以用来解决**连续状态下离散动作**的问题。

## 7.2 `CartPole`环境

下图所示的车杆环境中，它的状态值就是连续的，动作值是离散的。

![img](https://hrl.boyuai.com/static/cartpole.e4a03ca5.gif)

在车杆环境中，杆的一端固定小车，小车作为智能体的任务是**通过左右移动保持车上的杆竖直**，游戏的终止条件如下：
- 倾斜度数过大
- 车子离初始位置左右的偏离程度过大
- 坚持时间到达 200 帧

智能体的状态是一个维数为4的向量，每一维都是连续的，动作是离散的，动作空间大小为 2，详见下面两张表。

<center>表1 智能体的状态空间</center>

| 维度 |     意义     |  最小值   |  最大值  |
| :--: | :----------: | :-------: | :------: |
|  0   |   车的位置   |  $-2.4$   |  $2.4$   |
|  1   |   车的速度   | $-\infty$ | $\infty$ |
|  2   |   杆的角度   | $-41.8°$  | $41.8°$  |
|  3   | 杆尖端的速度 | $-\infty$ | $\infty$ |

<center>表2 智能体的动作空间</center>

| 标号 |     动作     |
| :--: | :----------: |
|  0   | 向左移动小车 |
|  1   | 向右移动小车 |

在游戏中每坚持一帧，智能体能获得分数为 1 的奖励，坚持时间越长，则最后的分数越高，坚持 200 帧即可获得最高的分数。

## 7.3 `DQN`算法

由于状态空间是连续的，车杆环境中的动作价值函数可以使用**函数拟合**(function approximation)的思想，即采用一个神经网络来表示函数。
- 若动作空间是连续的，神经网络的输入是状态和动作，然后输出一个标量，表示在状态下采取动作能获得的价值；
- 若动作空间是离散的，除了可以采取动作连续情况下的做法，我们还可以只将状态输入到神经网络中，使其同时输出每一个动作的值。

通常`DQN`(包括Q-learning)只能处理动作离散的情况，因为在函数的更新过程中有 $\mathop{max}_{a}$ 这一操作。假设神经网络用来拟合函数 $Q$ 的参数是 $\omega$，即每一个状态下所有可能动作的值我们都能表示为 $Q_{\omega}(s, a)$。我们称用于拟合函数 $Q$ 的神经网络为**Q网络**。

![img](https://hrl.boyuai.com/static/640.46b13e89.png)

损失函数的建立可以先回顾下 $Q-learning$ 的更新规则——采用时序差分来更新动作价值函数 $Q(s, a)$：
$$
Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \mathop{max}_{a^{\prime} \in \mathcal{A}}{Q(s^{\prime}, a^{\prime})} - Q(s, a)]
$$
于是，对于一组数据 $\{(s_i, a_i, r_i, s^{\prime}_i)\}$，我们可以将Q网络的损失函数构造为均方误差的形式：
$$
\omega^{*} = \mathop{argmin}_{\omega}{\frac{1}{2N} \sum_{i=1}^N[Q_{\omega}(s_i, a_i) - (r_i + \gamma \mathop{max}_{a^{\prime}}{Q_{\omega}(s^{\prime}_i, a^{\prime})})}]^2
$$
我们就可以将 $Q-learning$ 扩展到神经网络形式——**深度 Q 网络**(deep Q network, DQN)算法。由于`DQN`是**离线策略算法**(offline)，因此我们在收集数据的时候可以使用一个 $\epsilon-greedy$ 策略来平衡探索与利用，将收集到的数据存储起来，在后续的训练中使用。`DQN`中还有两个非常重要的模块——**经验回放**和**目标网络**，它们能够帮助`DQN`取得稳定、出色的性能。

### 7.3.1 经验回放

在一般的有监督学习中，假设训练数据是独立同分布($i.i.d$)的，我们每次训练神经网络的时候从训练数据中随机采样一个或若干个数据来进行梯度下降，随着学习的不断进行，每一个训练数据会被使用多次。在原来的 $Q-learning$ 算法中，每一个数据只会用来更新一次值。为了更好地将 $Q-learning$ 和深度神经网络结合，`DQN`算法采用了**经验回放**(experience replay)方法，具体做法为维护一个**回放缓冲区(buffer)**，将每次从环境中**采样**得到的四元组数据 $(s, a, r, s^{\prime})$ 存储到回放缓冲区中，训练Q网络的时候再从回放缓冲区中随机采样若干数据来进行训练。这么做可以起到以下两个作用：

- **使样本满足独立假设**：在MDP中交互采样得到的数据本身不满足独立假设，因为存在时序相关。非独立同分布的数据对训练神经网络有很大的影响，会使神经网络拟合到最近训练的数据上。采用经验回放可以打破样本之间的相关性，让其满足独立假设。
- **提高样本效率**。每一个样本可以被使用多次，十分适合深度神经网络的梯度学习。

### 7.3.2 目标网络

`DQN`算法的目标是让 $Q_{\omega}(s, a)$ 逼近 $r + \gamma \mathop{max}_{a^{\prime}}{Q_{\omega}(s^{\prime}, a^{\prime})}$，由于TD误差目标本身就包含神经网络的输出，因此在更新网络参数的同时目标也在不断地改变，即 $\mathop{max}_{a^{\prime}}{Q_{\omega}(s^{\prime}, a^{\prime})}$ 不断改变，这非常容易**造成神经网络训练的不稳定性**。为了解决这一问题，`DQN`采用了**目标网络**(target network)的思想：既然训练过程中Q网络的不断更新会导致目标不断发生改变，不如暂时先将TD目标中的Q网络固定住。为了实现这一思想，我们需要利用两套Q网络：目标网络和训练网络。

- 原来的训练网络 $Q_{\omega}(s, a)$，用于计算原来的损失函数 $\frac{1}{2}[Q_{\omega}(s, a) - (r+\gamma \mathop{max}_{a^{\prime}}{Q_{\omega^{-}}(s^{\prime}, a^{\prime})})]^2$ 中的 $Q_{\omega}(s, a)$ 项，并且使用正常梯度下降方法来进行更新；
- 目标网络 $Q_{\omega^{-}}(s, a)$，用于计算原先损失函数 $\frac{1}{2}[Q_{\omega}(s, a) - (r + \gamma \mathop{max}_{a^{\prime}}{Q_{\omega^{-}}(s^{\prime}, a^{\prime})})]^2$ 中的项，其中 $\omega^{-}$ 表示目标网络中的参数。

为了让更新目标更稳定，**目标网络并不会每一步都更新**。具体而言，目标网络使用训练网络的一套较旧的参数，训练网络 $Q_{\omega}(s, a)$ 在训练中的每一步都会更新，而目标网络的参数每隔 $C$ 步才会与训练网络同步一次，即 $\omega^{-} \leftarrow \omega$。

`DQN`算法的伪代码如下：

使用随机网络参数 $\omega$ 初始化网络 $Q_{\omega}(s, a)$

复制相同的参数 $\omega^{-} \leftarrow \omega$ 来初始化目标网络 $Q_{\omega^{-}}$

初始化经验回放池 $R$

$for$ 序列 $e=1 \rightarrow E \space do$ :

​	获取环境初始状态 $s_1$

​	$for$ 时间步$t=1 \rightarrow T \space do$ :

​		根据当前网络 $Q_{\omega}(s, a)$ 以 $\epsilon-greedy$ 策略选择动作 $a_t$

​		执行动作 $a_t$，获得回报 $r_t$，环境状态变为 $s_{t+1}$

​		将 $(s_t, a_t, r_t, s_{t+1})$ 存储进回放池 $R$ 中

​		若 $R$ 中数据足够，从 $R$ 中采样 $N$ 个数据 $\{(s_i, a_i, r_i, s_{i+1})\}_{i=1,\cdots,N}$

​		对每个数据，用目标网络计算 $y_i = r_i + \gamma \mathop{max}_{a}{Q_{\omega^{-}}(s_{i+1}, a)}$

​		最小化目标损失 $L = \frac{1}{N}{\sum_{i}{(y_i - Q_{\omega}(s_i, a_i))^2}}$ ，以此更新当前网络 $Q_{\omega}$

​		更新目标网络 $Q_{\omega^{-}}$

​	$end \space for$

$end \space for$

## 7.4 `DQN`代码实践

以下实现`DQN`代码：

```python
import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils

# 定义经验回放池
class ReplayBuffer:
    '''
    经验回放池
    '''
    def __init__(self, capacity):
        '''
        使用双向队列存储
        '''
        self.buffer = collections.deque(maxlen = capacity)

    def add(self, state, action, reward, next_state, done):
        '''
        将数据加入buffer
        '''
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        '''
        采样一个batch_size
        '''
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions) # zip函数返回元组，例如state元组中包括采样的所有的state
        
        return np.array(state), action, reward, np.array(next_state), done
    
    def size(self):
        '''
        经验回放池的大小
        '''
        return len(self.buffer)

# 定义只有一个隐藏层的Q网络
class Qnet(torch.nn.Module):
    '''
    Q网络
    '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# DQN算法
class DQN:
    '''
    DQN算法

    输入:
    :param state_dim: 状态维度
    :param hidden_dim: 隐藏层维度
    :param action_dim: 动作维度
    :param learning_rate: 学习率
    :param gamma: 折扣因子
    :param epsilon: epsilon-greedy策略参数
    :param target_update: 目标网络更新频率
    :param device: 训练设备
    '''
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate,
                gamma, epsilon, target_update, device):
        self.action_dim = action_dim # 动作维度
        self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device) # Q网络

        # 目标网络
        self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)

        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma # 折扣因子
        self.epsilon = epsilon # epsilon-greedy策略
        self.target_update = target_update # 目标网络更新频率
        self.count = 0 # 记录更新次数
        self.device = device # 设备

    # 采样中采用epsilon-greedy策略
    def take_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device) # 状态转化为torch的tensor
            action = self.q_net(state).argmax().item() # 贪婪

        return action
    
    # 更新
    def update(self, transition_dict):
        # 利用采样的batch数据进行更新
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards']).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # 采用Q-learning进行更新
        # 利用q_net计算q值
        q_values = self.q_net(states).gather(1, actions) # q值
        # gather(dim, index):gather函数的作用是沿着dim，按照index索引
        
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - done) # 计算各个状态的目标q值

        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets)) # 采用均方误差
        self.optimizer.zero_grad() # 梯度置零
        dqn_loss.backward() # 梯度反传
        self.optimizer.step() # 更新参数

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict()) # 同步参数

        self.count += 1

if __name__ == "__main__":
    # 设定基本参数
    lr = 2e-3
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 10
    buffer_size = 10000
    minimal_size = 500 #q网络训练所需的最少样本数
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # gym环境设定
    env_name = 'CartPole-v1'
    env = gym.make(env_name, render_mode='rgb_array')

    # 设定随机数种子
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # 经验回放池
    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()[0]
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ , _ = env.step(action) # 注意gym升级后，env.step返回observation, reward, terminated, truncated, info
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward

                    # 当buffer数据的数量超过一定值后,才进行Q网络训练
                    if replay_buffer.size() > minimal_size:
                        # 采样一个batch的数据
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        # 一个batch数据存入字典
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)

                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return': '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    # 绘制每个序列的回报
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format(env_name))
    plt.show()

    # 计算序列奖励的滑动平均
    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format(env_name))
    plt.show()
```

## 7.5 将图像作为输入的`DQN`算法

在一些视频游戏中，智能体并**不能直接获取这些状态信息**，而只能直接获取屏幕中的图像。我们可以将卷积层加入其网络结构以提取图像特征，最终实现以图像为输入的强化学习。

```python
class ConvolutionalQnet(torch.nn.Module):
    ''' 加入卷积层的Q网络 '''
    def __init__(self, action_dim, in_channels=4):
        super(ConvolutionalQnet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = torch.nn.Linear(7 * 7 * 64, 512)
        self.head = torch.nn.Linear(512, action_dim)

    def forward(self, x):
        x = x / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x))
        return self.head(x)
```

