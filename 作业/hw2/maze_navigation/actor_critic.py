import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from maze_navigation_utils import MazeEnv, moving_average

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
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        # self.action_bound = action_bound # 动作界限
        self.fc = torch.nn.Linear(state_dim, hidden_dim)
        # self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

        # 均值和方差
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # x = F.relu(self.fc1(x))
        # return torch.tanh(self.fc2(x)) * self.action_bound
        x = F.relu(self.fc(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        std = torch.exp(log_std)

        return mean, std

# 状态值函数网络
class ValueNet(torch.nn.Module):
    '''
    状态价值网络，输出状态价值

    输入:
    state_dim:状态空间维度
    hidden_dim:隐藏层维度
    '''
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Actor-Critic
class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, action_lb, action_ub,
                 actor_lr, critic_lr, gamma, device):
        # 策略网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)  # 价值网络
        # 策略网络优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)  # 价值网络优化器
        
        self.action_lb = action_lb # 动作下界
        self.action_ub = action_ub # 动作上界
        self.gamma = gamma # 折扣因子
        self.device = device # 设备

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device) # 状态变成tensor
        action_distribution = torch.distributions.normal.Normal(*self.actor(state)) # 动作空间分布
        action_unclamped = action_distribution.sample() # 未裁剪的动作

        return action_unclamped.clamp(self.action_lb, self.action_ub).detach().numpy()
        # action = self.actor(state).detach().numpy()

        # return action

    def update(self, transition_dict):
        # 记录数据
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # # 时序差分目标
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)  # 时序差分误差
        mean, std = self.actor(states)
        normal = torch.distributions.normal.Normal(mean, std)
        log_probs = normal.log_prob(actions)
        actor_loss = torch.mean(-log_probs * td_delta.detach())

        # # 均方误差损失函数
        # critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
        # self.actor_optimizer.zero_grad()
        # self.critic_optimizer.zero_grad()
        # actor_loss.backward()  # 计算策略网络的梯度
        # critic_loss.backward()  # 计算价值网络的梯度
        # self.actor_optimizer.step()  # 更新策略网络的参数
        # self.critic_optimizer.step()  # 更新价值网络的参数

        # actor误差
        # actor_loss = -self.critic(states)
        # actor_loss = torch.mean(actor_loss)

        # critic误差
        # td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()  # 计算策略网络的梯度
        critic_loss.backward()  # 计算价值网络的梯度
        self.actor_optimizer.step()  # 更新策略网络的参数
        self.critic_optimizer.step()  # 更新价值网络的参数

if __name__ == '__main__':
    # 环境设置
    state_lb = -0.5
    state_ub = 0.5
    x_terminal = 0.5
    y_terminal = 0.5
    epsilon = 1e-2

    state_dim = 2
    hidden_dim = 4
    action_dim = 2
    action_lb = -0.1
    action_ub = 0.1
    action_bound = min(abs(action_lb), abs(action_ub))

    # 智能体参数设置
    actor_lr = 1e-3
    critic_lr = 1e-3
    num_episodes = 1000
    gamma = 0.98
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    torch.manual_seed(12)
    np.random.seed(12)

    env = MazeEnv(state_lb, state_ub, x_terminal, y_terminal, epsilon)
    agent = ActorCritic(state_dim, hidden_dim, action_dim, action_lb, action_ub,
                        actor_lr, critic_lr, gamma, device)

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
    plt.title("Actor-Critic on maze navigation")
    plt.xlabel("episode")
    plt.ylabel("return")
    plt.show()
    
    mv_return = moving_average(return_list, 19)
    plt.plot(mv_return)
    plt.title("Actor-Critic on maze navigation")
    plt.xlabel("episode")
    plt.ylabel("average return on last 20 episodes")
    plt.show()
