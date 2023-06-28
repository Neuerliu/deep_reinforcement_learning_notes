import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

import torch
import torch.nn.functional as F
from torch import nn

from maze_navigation_utils import MazeEnv, moving_average, kl_divergence, flat_grad, flat_hessian, flat_params

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
        self.mean_layer.weight.data.mul_(0.1)
        self.mean_layer.bias.data.mul_(0.0)
        self.log_std_layer.weight.data.mul_(0.1)
        self.log_std_layer.bias.data.mul_(0.0)
        # self.log_std_layer.weight.data.fill_(0.0)
        # self.log_std_layer.bias.data.fill_(0.0)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        # log_std = torch.zeros_like(mean)
        std = torch.exp(log_std)

        return mean, std

class NaturalPolicyGradient:
    '''
    自然策略梯度法
    '''
    def __init__(self, state_dim, hidden_dims, action_dim, action_lb, 
                 action_ub, gamma, delta, device):
        self.action_lb = action_lb # 动作下界
        self.action_ub = action_ub # 动作上界
        self.policy_net = PolicyNet(state_dim, hidden_dims, action_dim).to(device) # 策略网络
        self.gamma = gamma # 折扣因子
        self.delta = delta # 自然梯度的约束参数
        self.device = device # 设备
        
    # 使用均值方差采取行动
    def take_action(self, state):
        self.policy_net.eval()
        state = torch.tensor(state, dtype=torch.float).to(self.device) # 状态变成tensor
        mean, std = self.policy_net(state)
        action_unclamped = torch.normal(mean, std) # 动作空间分布
        
        return list(action_unclamped.clamp(self.action_lb, self.action_ub).detach().numpy())

    # 计算fisher信息矩阵
    def fisher_vector_product(self, state_tensor, p):
        p.detach()
        kl = kl_divergence(self.policy_net, self.policy_net, state_tensor)
        kl = kl.mean()
        kl_grad = torch.autograd.grad(kl, self.policy_net.parameters(), create_graph=True)
        kl_grad = flat_grad(kl_grad)

        kl_grad_p = (kl_grad * p).sum()
        kl_hessian_p = torch.autograd.grad(kl_grad_p, self.policy_net.parameters())
        kl_hessian_p = flat_hessian(kl_hessian_p)

        return kl_hessian_p + 0.1 * p
    
    # 使用共轭梯度法求解梯度
    def conjugate_gradient(self, state_tensor, grad_tensor, nsteps, delta):
        x = torch.zeros(grad_tensor.size())
        r = grad_tensor.clone()
        p = grad_tensor.clone()
        rdotr = torch.dot(r, r)
        for _ in range(nsteps):
            _Avp = self.fisher_vector_product(state_tensor, p)
            alpha = rdotr / torch.dot(p, _Avp)
            x += alpha * p
            r -= alpha * _Avp
            new_rdotr = torch.dot(r, r)
            betta = new_rdotr / rdotr
            p = r + betta * p
            rdotr = new_rdotr
            if rdotr < delta:
                break

        return x

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

        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        mean, std = self.policy_net(state_tensor)
        normal = torch.distributions.normal.Normal(mean, std)
        log_probs = normal.log_prob(action_tensor).unsqueeze(-1)

        loss = (log_probs * advantages).mean()

        # 计算损失的梯度并展平
        loss_grad = torch.autograd.grad(loss, self.policy_net.parameters())
        loss_grad = flat_grad(loss_grad)
        step_dir = self.conjugate_gradient(state_tensor, loss_grad.data, 50, self.delta) # 计算梯度更新步长
        
        # 更新参数
        params = flat_params(self.policy_net)
        new_params = params - 0.5 * step_dir
        self.update_model(new_params)

    # 更新模型参数
    def update_model(self, new_params):
        index = 0
        for params in self.policy_net.parameters():
            params_length = len(params.view(-1))
            new_param = new_params[index: index + params_length]
            new_param = new_param.view(params.size())
            params.data.copy_(new_param)
            index += params_length

if __name__ == '__main__':
    # 设置基本参数
    state_lb = -0.5
    state_ub = 0.5
    x_terminal = 0.5
    y_terminal = 0.5
    epsilon = 1e-3

    state_dim = 2
    hidden_dims = [8, 4]
    action_dim = 2
    action_lb = -0.1
    action_ub = 0.1

    # 定义智能体相关参数
    gamma = 0.95
    num_episodes = 50
    delta = 1e-5
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 定义环境和智能体
    env = MazeEnv(state_lb, state_ub, x_terminal, y_terminal, epsilon)
    agent = NaturalPolicyGradient(state_dim, hidden_dims, action_dim, action_lb, action_ub, gamma, delta, device)

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
    plt.title("natural policy gradient on maze navigation")
    plt.xlabel("episode")
    plt.ylabel("return")
    plt.show()
    
    mv_return = moving_average(return_list, 99)
    plt.plot(mv_return)
    plt.title("natural policy gradient on maze navigation")
    plt.xlabel("episode")
    plt.ylabel("average return on last 100 episodes")
    plt.show()
