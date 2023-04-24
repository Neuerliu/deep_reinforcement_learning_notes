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
```

