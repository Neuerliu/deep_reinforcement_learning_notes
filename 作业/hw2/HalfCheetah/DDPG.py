import random
import gym
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from halfcheetah_utils import moving_average, ReplayBuffer

# 策略网络
class PolicyNet(torch.nn.Module):
    '''
    策略网络，输出动作

    输入:
    state_dim:状态空间维度
    hidden_dim:隐藏层维度
    action_dim:动作空间维度
    action_bound:状态的上下界
    '''
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound  # action_bound是环境可以接受的动作最大值

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)) * self.action_bound

# Q值网络
class QValueNet(torch.nn.Module):
    '''
    拟合状态动作对应的Q值网络

    输入:
    state_dim:状态空间维度
    hidden_dim:隐藏层维度
    action_dim:动作空间维度
    action_bound:状态的上下界
    '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.action_dim = action_dim
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1) # 拼接状态和动作
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)
    
# DDPG算法
class DDPG:
    '''
    DDPG算法，采用Actor-Critic算法

    输入:
    state_dim:状态空间维度
    hidden_dim:隐藏层维度
    action_dim:动作空间维度
    action_bound:状态的上下界
    '''
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, 
                 sigma, actor_lr, critic_lr, tau, gamma, device):
        # 仿照改进的DQN，采用两套网络，即目标网络和训练网络
        # 训练网络
        # 通过actor输出一个动作
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        # 使用critic输出该动作的Q值
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        # 目标网络
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.target_critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)

        # 初始化目标价值网络并设置和价值网络相同的参数
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化目标策略网络并设置和策略相同的参数
        self.target_actor.load_state_dict(self.actor.state_dict())

        # 设置优化函数
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma # 折扣因子
        self.sigma = sigma  # 高斯噪声的标准差,均值直接设为0
        self.tau = tau  # 目标网络软更新参数
        self.action_dim = action_dim # 状态空间维度
        self.device = device # 设备

    # 行动函数
    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = self.actor(state).detach().numpy()
        # 给动作添加噪声，增加探索
        action = action + self.sigma * np.random.randn(self.action_dim)
        return action

    # 软更新
    def soft_update(self, net, target_net):
        # 采用软更新
        # w- <- tau * w + (1-tau) * w-
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    # 更新网络参数
    def update(self, transition_dict):
        # 整理状态、动作、奖励、下一个状态以及是否为终止状态
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # 更新critic网络
        # 计算当前的目标Q值和估计Q值之间的误差
        next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新actor网络
        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
        self.soft_update(self.critic, self.target_critic)  # 软更新价值网络


if __name__ == '__main__':
    # 环境基本参数设置
    env = gym.make('HalfCheetah-v2')
    state_dim = env.observation_space.shape[0]
    hidden_dim = 32
    action_dim = env.action_space.shape[0]
    action_lb = env.action_space.low[0]
    action_ub = env.action_space.high[0]
    action_bound = min(abs(action_lb), abs(action_ub))

    # 智能体基本参数设置
    actor_lr = 1e-3
    critic_lr = 1e-3
    num_episodes = 1000
    gamma = 0.98
    tau = 0.005  # 软更新参数
    buffer_size = 10000
    minimal_size = 1000
    batch_size = 64
    sigma = 0.01  # 高斯噪声标准差
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 随机数种子
    random.seed(12)
    np.random.seed(12)
    torch.manual_seed(12)

    replay_buffer = ReplayBuffer(buffer_size)
    agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, 
                 sigma, actor_lr, critic_lr, tau, gamma, device)

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                done = False
                count = 0
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    count += 1

                    # 当buffer足够大，再开始采样
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 
                                           'actions': b_a, 
                                           'next_states': b_ns, 
                                           'rewards': b_r, 
                                           'dones': b_d
                                           }
                        agent.update(transition_dict)

                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)

    plt.plot(return_list)
    plt.title("DDPG on Half Cheetah")
    plt.xlabel("episode")
    plt.ylabel("return")
    plt.show()
    
    mv_return = moving_average(return_list, 19)
    plt.plot(mv_return)
    plt.title("DDPG on Half Cheetah")
    plt.xlabel("episode")
    plt.ylabel("average return on last 20 episodes")
    plt.show()
