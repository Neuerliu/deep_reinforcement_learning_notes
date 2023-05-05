import random
import gym
# import gymnasium as gym
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
        if np.random.random() < self.epsilon:
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
    env = gym.make(env_name) # , render_mode='rgb_array'

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
                    next_state, reward, done, _ , __ = env.step(action) # 注意gym升级后，env.step返回observation, reward, terminated, truncated, info
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
