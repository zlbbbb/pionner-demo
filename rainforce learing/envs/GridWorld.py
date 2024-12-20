import numpy as np
import gym
from gym import spaces
import matplotlib

matplotlib.use('TkAgg')  # 使用TkAgg后端
import matplotlib.pyplot as plt


class GridWorld5x5(gym.Env):
    """
    5x5网格世界环境
    特点：
    - 固定5x5大小
    - 包含起点、终点和障碍物
    - 四个移动方向
    - 简单的奖励机制
    """

    def __init__(self):
        super(GridWorld5x5, self).__init__()

        # 设置固定的网格大小为5x5
        self.size = 5

        # 定义动作空间: 0=上, 1=下, 2=左, 3=右
        self.action_space = spaces.Discrete(4)

        # 定义状态空间
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.size, self.size), dtype=np.float32)

        # 定义动作映射
        self.action_to_direction = {0: (-1, 0),  # 上
            1: (1, 0),  # 下
            2: (0, -1),  # 左
            3: (0, 1)  # 右
        }

        # 定义元素的值
        self.EMPTY = 0.0  # 空格子
        self.AGENT = 0.5  # 智能体
        self.GOAL = 1.0  # 目标
        self.OBSTACLE = -1.0  # 障碍物

        # 定义奖励
        self.STEP_REWARD = -0.1  # 每步的奖励
        self.OBSTACLE_REWARD = -1.0  # 撞到障碍物的奖励
        self.GOAL_REWARD = 1.0  # 到达目标的奖励

        # 初始化显示
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))

        # 初始化环境
        self.reset()

    def reset(self):
        """重置环境到初始状态"""
        # 创建空网格
        self.grid = np.zeros((self.size, self.size))

        # 设置固定的起点（左上角）
        self.agent_pos = [0, 0]

        # 设置固定的终点（右下角）
        self.goal_pos = [self.size - 1, self.size - 1]

        # 设置固定的障碍物位置
        #self.obstacles = [[1, 1], [2, 2], [3, 1]]
        self.obstacles = []
        while len(self.obstacles) < 3:  # 放置3个随机障碍物
            pos = [np.random.randint(0, self.size), np.random.randint(0, self.size)]
            if pos != self.agent_pos and pos != self.goal_pos:
                self.obstacles.append(pos)
        # 更新网格状态
        self._update_grid()

        return self.grid

    def _update_grid(self):
        """更新网格状态"""
        # 清空网格
        self.grid.fill(self.EMPTY)

        # 放置智能体
        self.grid[self.agent_pos[0], self.agent_pos[1]] = self.AGENT

        # 放置目标
        self.grid[self.goal_pos[0], self.goal_pos[1]] = self.GOAL

        # 放置障碍物
        for obs in self.obstacles:
            self.grid[obs[0], obs[1]] = self.OBSTACLE

    def step(self, action):
        """执行一个动作"""
        # 获取动作对应的方向
        direction = self.action_to_direction[action]

        # 计算新位置
        new_pos = [self.agent_pos[0] + direction[0], self.agent_pos[1] + direction[1]]

        # 初始化奖励和结束标志
        reward = self.STEP_REWARD
        done = False

        # 检查是否在网格内
        if (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):

            # 检查是否撞到障碍物
            if new_pos in self.obstacles:
                reward = self.OBSTACLE_REWARD
            else:
                # 更新位置
                self.agent_pos = new_pos

                # 检查是否到达目标
                if self.agent_pos == self.goal_pos:
                    reward = self.GOAL_REWARD
                    done = True
        else:
            # 撞墙
            reward = self.OBSTACLE_REWARD

        # 更新网格
        self._update_grid()

        return self.grid, reward, done, {}

    def render(self, mode='human'):
        """渲染环境"""
        if mode == 'human':
            self.ax.clear()

            # 显示网格
            img = self.ax.imshow(self.grid, cmap='RdYlGn')

            # 添加网格线
            self.ax.grid(True)

            # 添加标题
            self.ax.set_title('5x5 Grid World')

            # 添加坐标标签
            for i in range(self.size):
                for j in range(self.size):
                    if self.grid[i, j] == self.AGENT:
                        text = 'A'
                    elif self.grid[i, j] == self.GOAL:
                        text = 'G'
                    elif self.grid[i, j] == self.OBSTACLE:
                        text = 'O'
                    else:
                        text = ''
                    self.ax.text(j, i, text, ha='center', va='center')

            # 更新显示
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.1)

    def close(self):
        """关闭环境"""
        plt.ioff()
        plt.close()


def test_environment():
    """测试环境"""
    # 创建环境
    env = GridWorld5x5()

    # 运行几个测试回合
    n_episodes = 3
    max_steps = 40

    try:
        for episode in range(n_episodes):
            state = env.reset()
            total_reward = 0

            print(f"\nEpisode {episode + 1}")
            env.render()

            for step in range(max_steps):
                # 随机选择动作
                action = env.action_space.sample()

                # 执行动作
                next_state, reward, done, _ = env.step(action)
                total_reward += reward

                # 显示信息
                print(f"Step {step + 1}: Action={action}, Reward={reward:.2f}")
                env.render()

                if done:
                    print(f"Episode finished after {step + 1} steps")
                    break

            print(f"Total reward: {total_reward:.2f}")

    finally:
        env.close()


if __name__ == "__main__":
    test_environment()