class CliffWalkingEnv:
    '''
    悬崖漫步环境
    '''
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol # 行数
        self.nrow = nrow # 列数
        # 状态转移函数
        self.P = self.createP() 
        # p[state][action] = [(p, next_state, reward, done)]
        # 对应(转移概率, 下一个状态, 奖励, 是否处于终止状态)
    
    # 状态转移函数
    def createP(self):
        # 初始化
        # 状态转移函数
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)] # P中48个状态，每一个状态可以采用4中动作
        # 动作初始化
        change = [
            [0, -1],
            [0, 1],
            [-1, 0],
            [1, 0]
        ] # 分别对应上、下、左、右

        # 遍历状态转移函数每个(s, a)，定义状态转移概率
        # 悬崖位于self.nrow - 1, 且在这一列中j=0代表起始位置，j=self.ncol-1代表终点位置
        for i in range(self.nrow): # 当前y
            for j in range(self.ncol): # 当前x
                # (i,j)代表此时位置
                for a in range(4):
                    # 此时采取动作a
                    # 1.如果位置在悬崖或者目标状态(终点)，则无法进行任何交互，任何动作奖励为0
                    if i == self.nrow - 1 and j > 0:
                        P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0, True)]
                        continue # 跳出循环

                    # 2.其他位置
                    # 限制x位置不能超出左右边界
                    next_x = min(self.ncol - 1, max(0, j + change[a][0]))
                    # 限制y位置不能超出边界
                    next_y = min(self.nrow - 1, max(0, i + change[a][1]))
                    next_state = next_y * self.ncol + next_x
                    # 非终点和悬崖位置奖励为-1
                    reward = -1
                    done = False

                    # 判断其他位置的下一个位置是否在悬崖或终点
                    if next_y == self.nrow - 1 and next_x > 0:
                        done = True
                        # 判断是否在终点
                        if next_x != self.ncol - 1:
                            # 在悬崖而非终点
                            reward = -100
                    P[i * self.ncol + j][a] = [(1, next_state, reward, done)]

        return P
