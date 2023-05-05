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
