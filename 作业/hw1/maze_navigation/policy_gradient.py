import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim

from maze_navigation_utils import MazeEnv, moving_average

# 定义策略网络
class PolicyNet(nn.Module):
    '''
    定义策略网络
    '''
    def __init__(self, state_dim, hidden_dims, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])

        # 均值和方差
        self.mean_layer = nn.Linear(hidden_dims[1], action_dim)
        self.log_std_layer = nn.Linear(hidden_dims[1], action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        std = torch.exp(log_std)

        return mean, std

# 策略梯度法
class PolicyGradient:
    '''
    策略梯度法
    '''
    def __init__(self, state_dim, hidden_dims, action_dim, action_lb, 
                 action_ub, learning_rate, gamma, device):
        self.action_lb = action_lb # 动作下界
        self.action_ub = action_ub # 动作上界
        self.policy_net = PolicyNet(state_dim, hidden_dims, action_dim).to(device) # 策略网络
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate) # 优化器
        self.gamma = gamma # 折扣因子
        self.device = device # 设备
        
    # 使用均值方差采取行动
    def take_action(self, state):
        self.policy_net.eval()
        state = torch.tensor(state, dtype=torch.float).to(self.device) # 状态变成tensor
        action_distribution = torch.distributions.normal.Normal(*self.policy_net(state)) # 动作空间分布
        action_unclamped = action_distribution.sample() # 未裁剪的动作

        return list(action_unclamped.clamp(self.action_lb, self.action_ub).numpy())

    # 更新网络
    def update(self, transition_dict):
        self.policy_net.train()
        # 获取序列的奖励、状态、动作
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        # 奖励、动作的tensor
        reward_tensor = torch.tensor(reward_list, dtype=torch.float)
        state_tensor = torch.tensor(state_list, dtype=torch.float)
        action_tensor = torch.tensor(action_list, dtype=torch.float)

        advantages = torch.zeros_like(reward_tensor).to(self.device)
        G = 0
        for i in reversed(range(len(reward_list))):
            G = reward_list[i] + self.gamma * G
            advantages[i] = G

        mean, std = self.policy_net(state_tensor)
        normal = torch.distributions.normal.Normal(mean, std)
        log_probs = normal.log_prob(action_tensor).unsqueeze(-1)

        loss = -(log_probs * advantages).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__ == '__main__':
    # 设置基本参数
    state_lb = -0.5
    state_ub = 0.5
    x_terminal = 0.5
    y_terminal = 0.5
    epsilon = 1e-2

    state_dim = 2
    hidden_dims = [8, 4]
    action_dim = 2
    action_lb = -0.1
    action_ub = 0.1

    torch.manual_seed(12)
    np.random.seed(12)

    # 定义智能体相关参数
    learning_rate = 1e-4
    gamma = 0.95
    num_episodes = 6000
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 定义环境和智能体
    env = MazeEnv(state_lb, state_ub, x_terminal, y_terminal, epsilon)
    agent = PolicyGradient(state_dim, hidden_dims, action_dim, action_lb, 
                            action_ub, learning_rate, gamma, device)

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'dones': []
                }
                state = env.reset()
                done = False
                count = 0
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done = env.step(action, count)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                    count += 1

                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    plt.plot(return_list)
    plt.title("policy gradient on maze navigation")
    plt.xlabel("episode")
    plt.ylabel("return")
    plt.show()
    
    mv_return = moving_average(return_list, 99)
    plt.plot(mv_return)
    plt.title("policy gradient on maze navigation")
    plt.xlabel("episode")
    plt.ylabel("average return on last 100 episodes")
    plt.show()
