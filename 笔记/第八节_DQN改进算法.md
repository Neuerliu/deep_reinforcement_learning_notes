# `DQN`改进算法

## 8.1 简介

本章将介绍两个对`DQN`算法的改进——`Double DQN`和`Dueling DQN`，这两个算法的实现非常简单，只需要在`DQN`的基础上稍加修改，它们能在一定程度上改善`DQN`的效果。

## 8.2 `Double DQN`算法

普通的`DQN`算法通常会导致对值的过高估计(overestimation)。传统`DQN`优化的TD误差目标为：
$$
r + \gamma \mathop{max}_{a^{\prime}}{Q_{\omega^{-}}(s^{\prime}, a^{\prime})}
$$
其中 $\mathop{max}_{a^{\prime}}{Q_{\omega^{-}}(s^{\prime}, a^{\prime})}$ 由目标网络 $Q_{\omega^{-}}$ 计算得到，我们可以改写成为如下形式：
$$
Q_{\omega^{-}}(s^{\prime}, \mathop{argmax}_{a^{\prime}}{Q_{\omega^{-}}(s^{\prime}, a^{\prime})})
$$
上式中 $max$ 的操作可以被拆分为两部分：

- 选取状态 $s^{\prime}$ 下的最优动作 $a^{*} = \mathop{argmax}_{a^{\prime}}{Q_{\omega^{-}}{(s^{\prime}, a^{\prime})}}$
- 计算最优动作对应的动作价值 $Q_{\omega^{-}}{(s^{\prime}, a^{*})}$

当这两部分采用同一套 $Q$ 网络进行计算时，每次得到的都是神经网络当前估算的所有动作价值中的最大值。考虑到通过神经网络估算的 $Q$ 值本身在某些时候会产生正向或负向的误差，在`DQN`的更新方式下**神经网络会将正向误差累积**。为了解决这一问题，`Double DQN`算法提出**利用两个独立训练的神经网络估算**。具体做法是将原有的 $\mathop{max}_{a^{\prime}}{Q_{\omega^{-}}(s^{\prime}, a^{\prime})}$ 更改为 $Q_{\omega^{-}}{(s^{\prime}, \mathop{argmax}_{a^{-}}{Q_{\omega}(s^{\prime}, a^{\prime})})}$ ，即利用一套神经网络 $Q_{\omega}$ 的输出选取价值最大的动作，但在使用该动作的价值时，用另一套神经网络 $Q_{\omega^{-}}$ 计算该动作的价值。即使其中一套神经网络的某个动作存在比较严重的过高估计问题，由于另一套神经网络的存在，这个动作最终使用的值不会存在很大的过高估计问题。

实际上，在`DQN`中本身就有两套 $Q$ 函数的神经网络——目标网络和训练网络，我们之前只用到目标网络计算 $\mathop{max}_{a^{\prime}}{Q_{\omega^{-}}(s^{\prime}, a^{\prime})}$ ，现在可以考虑使用训练网络选取动作，即将训练网络作为 $Q_{\omega}$ 。此时可以直接写出`Double DQN`的优化目标`TD error`：
$$
r + \gamma Q_{\omega^{-}}(s^{\prime}, \mathop{argmax}_{a^{\prime}}{Q_{\omega}(s^{\prime}, a^{\prime})})
$$

## 8.3 `Double DQN`代码实践

`DQN`与`Double DQN`的差别只是在于计算状态 $s^{\prime}$ 下 $Q$ 值时如何选取动作：
