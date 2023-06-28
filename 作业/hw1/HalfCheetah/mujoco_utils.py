import numpy as np
import torch

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) # 在index=0处插入0，并计算累积值
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

# 计算kl散度
def kl_divergence(new_agent, old_agent, state_tensor):
    '''
    计算连续动作空间下两个智能体动作分布的kl散度
    '''
    new_mean, new_std = new_agent(state_tensor)
    old_mean, old_std = old_agent(state_tensor)
    old_mean, old_std = old_mean.detach(), old_std.detach()

    kl = torch.log(old_std) - torch.log(new_std) + (old_std.pow(2) + (old_mean - new_mean).pow(2)) / (2.0 * new_std.pow(2) + 1e-10) - 0.5
    return kl.sum(1, keepdim=True)

# 展平梯度
def flat_grad(grads):
    grad_flatten = []
    for grad in grads:
        grad_flatten.append(grad.view(-1))
    grad_flatten = torch.cat(grad_flatten)

    return grad_flatten

# 展平Hessian矩阵
def flat_hessian(hessians):
    hessians_flatten = []
    for hessian in hessians:
        hessians_flatten.append(hessian.contiguous().view(-1))
    hessians_flatten = torch.cat(hessians_flatten).data
    
    return hessians_flatten

# 展平参数
def flat_params(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    params_flatten = torch.cat(params)
    
    return params_flatten
