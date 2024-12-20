import numpy as np
import gym
from gym import spaces
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class GridWorld5x5(gym.Env):
    """
    5x5网格世界环境
    使用plot方法实现可视化
    动作空间：上、右、下、左、保持不动
    """

    def __init__(self):
        super(GridWorld5x5, self).__init__()

        # 基础设置
        self.size = 5

        # 动作空间：0=上, 1=右, 2=下, 3=左, 4=保持不动
        self.action_space = spaces.Discrete(5)

        # 状态空间
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.size, self.size), dtype=np.float32)

        # 动作映射
        self.action_to_direction = {0: (-1, 0),  # 上
            1: (0, 1),  # 右
            2: (1, 0),  # 下
            3: (0, -1),  # 左
            4: (0, 0)  # 保持不动
        }

        # 动作名称
        self.action_names = {0: "上", 1: "右", 2: "下", 3: "左", 4: "保持不动"}

        # 初始化显示
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))

        # 初始化环境
        self.reset()

    def reset(self):
        """重置环境"""
        # 初始化智能体位置（左上角）
        self.agent_pos = [0, 0]

        # 初始化目标位置（右下角）
        self.goal_pos = [self.size - 1, self.size - 1]

        # 初始化障碍物
        self.obstacles = [[1, 1], [2, 2], [3, 1], [1, 3]]

        return self._get_state()

    def _get_state(self):
        """获取当前状态"""
        state = np.zeros((self.size, self.size))

        # 设置智能体位置
        state[self.agent_pos[0], self.agent_pos[1]] = 1

        return state

    def step(self, action):
        """执行动作"""
        # 获取移动方向
        direction = self.action_to_direction[action]

        # 计算新位置
        new_pos = [self.agent_pos[0] + direction[0], self.agent_pos[1] + direction[1]]

        # 初始化奖励和结束标志
        reward = -0.1  # 基础移动奖励
        done = False
        info = {"action_name": self.action_names[action]}

        # 处理保持不动
        if action == 4:
            reward = -0.2
        else:
            # 检查是否在网格内
            if (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):

                # 检查是否撞到障碍物
                if new_pos in self.obstacles:
                    reward = -1.0
                else:
                    # 更新位置
                    self.agent_pos = new_pos

                    # 检查是否到达目标
                    if self.agent_pos == self.goal_pos:
                        reward = 1.0
                        done = True
            else:
                reward = -1.0  # 撞墙惩罚

        return self._get_state(), reward, done, info

    def render(self, mode='human'):
        """使用plot方法渲染环境"""
        if mode == 'human':
            self.ax.clear()

            # 设置坐标轴范围
            self.ax.set_xlim(-0.5, self.size - 0.5)
            self.ax.set_ylim(-0.5, self.size - 0.5)

            # 绘制网格线
            for i in range(self.size + 1):
                self.ax.plot([i - 0.5, i - 0.5], [-0.5, self.size - 0.5], 'k-', lw=1)
                self.ax.plot([-0.5, self.size - 0.5], [i - 0.5, i - 0.5], 'k-', lw=1)

            # 绘制障碍物（红色方块）
            for obs in self.obstacles:
                self.ax.add_patch(plt.Rectangle((obs[1] - 0.5, obs[0] - 0.5), 1, 1, facecolor='red', alpha=0.3))

            # 绘制目标（绿色方块）
            self.ax.add_patch(
                plt.Rectangle((self.goal_pos[1] - 0.5, self.goal_pos[0] - 0.5), 1, 1, facecolor='green', alpha=0.3))

            # 绘制智能体（蓝色圆形）
            self.ax.plot(self.agent_pos[1], self.agent_pos[0], 'bo', markersize=20, alpha=0.5)

            # 添加网格坐标
            for i in range(self.size):
                for j in range(self.size):
                    self.ax.text(j, i, f'({i},{j})', ha='center', va='center', fontsize=8)

            # 设置标题和标签
            self.ax.set_title('5x5 Grid World')
            self.ax.set_xlabel('X轴')
            self.ax.set_ylabel('Y轴')

            # 保持坐标轴比例相等
            self.ax.set_aspect('equal')

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
    env = GridWorld5x5()
    n_episodes = 2
    max_steps = 20

    try:
        for episode in range(n_episodes):
            state = env.reset()
            total_reward = 0

            print(f"\n=== Episode {episode + 1} ===")
            env.render()

            for step in range(max_steps):
                # 随机选择动作
                action = env.action_space.sample()

                # 执行动作
                next_state, reward, done, info = env.step(action)
                total_reward += reward

                # 显示信息
                print(f"Step {step + 1}: {info['action_name']}, Reward={reward:.2f}")
                env.render()

                if done:
                    print(f"目标达成！Episode在{step + 1}步后完成")
                    break

            print(f"总奖励: {total_reward:.2f}")

    finally:
        env.close()


if __name__ == "__main__":
    test_environment()