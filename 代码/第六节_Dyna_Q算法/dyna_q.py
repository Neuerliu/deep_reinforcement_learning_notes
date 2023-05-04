import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import time
from dyna_q_utils import DynaQ, DynaQ_CliffWalking, CliffWalkingEnv

if __name__ == '__main__':
    # 展示结果
    np.random.seed(0)
    random.seed(0)
    n_planning_list = [0, 2, 20] # 不同n_planning
    for n_planning in n_planning_list:
        print('Q-planning步数为：%d' % n_planning)
        time.sleep(0.5)
        return_list = DynaQ_CliffWalking(n_planning)
        episodes_list = list(range(len(return_list)))
        plt.plot(episodes_list, return_list, label=str(n_planning) + ' planning steps')

    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Dyna-Q on {}'.format('Cliff Walking'))
    plt.show()
