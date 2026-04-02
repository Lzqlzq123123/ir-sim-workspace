"""
训练可视化回调模块

提供训练过程中的实时可视化功能

控制型可视化: 显示车辆运动状态、激光雷达
规划型可视化: 显示车辆运动状态 + 规划轨迹 + 目标点
"""

import os
import numpy as np
from typing import Optional
from stable_baselines3.common.callbacks import BaseCallback


class ControlVisualizeCallback(BaseCallback):
    """
    控制型训练可视化回调

    定期执行可视化评估，显示:
    - 车辆运动轨迹
    - 激光雷达扫描
    - 障碍物
    - 目标位置
    """

    def __init__(
        self,
        visualize_freq: int = 5000,
        n_visualize_steps: int = 200,
        render_delay: float = 0.02,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.visualize_freq = visualize_freq
        self.n_visualize_steps = n_visualize_steps
        self.render_delay = render_delay

    def _on_step(self) -> bool:
        if self.n_calls % self.visualize_freq == 0 and self.n_calls > 0:
            self._visualize_episode()
        return True

    def _visualize_episode(self) -> None:
        """执行可视化回合"""
        import matplotlib.pyplot as plt

        print(f"\n{'='*50}")
        print(f"[控制型可视化] Step {self.n_calls}")
        print(f"{'='*50}")

        # 创建可视化环境
        from control_gym.env import ControlEnv
        viz_env = ControlEnv(env_yaml="env.yaml", render_mode="human")
        obs, _ = viz_env.reset()

        total_reward = 0
        step_count = 0

        # 记录轨迹用于后续分析
        trajectory = []

        for step in range(self.n_visualize_steps):
            action, _ = self.model.predict(obs, deterministic=True)

            # 记录位置
            robot = viz_env.env.robot
            trajectory.append([
                float(robot.state[0, 0]),
                float(robot.state[1, 0]),
                float(robot.state[2, 0])
            ])

            obs, reward, terminated, truncated, info = viz_env.step(action)
            total_reward += reward
            step_count += 1

            viz_env.render()
            plt.pause(self.render_delay)

            if terminated or truncated:
                break

        # 结果统计
        result = "✓ 到达目标" if info.get("arrive") else "✗ 碰撞" if info.get("collision") else "○ 超时"
        print(f"结果: {result}")
        print(f"步数: {step_count}")
        print(f"总奖励: {total_reward:.1f}")
        print(f"{'='*50}\n")

        viz_env.close()


class PlanningVisualizeCallback(BaseCallback):
    """
    规划型训练可视化回调

    定期执行可视化评估，显示:
    - 车辆运动轨迹
    - 规划的参考轨迹（绿色虚线）
    - 目标位置（红色点）
    - 激光雷达扫描
    - 障碍物
    """

    def __init__(
        self,
        visualize_freq: int = 5000,
        n_visualize_steps: int = 200,
        render_delay: float = 0.02,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.visualize_freq = visualize_freq
        self.n_visualize_steps = n_visualize_steps
        self.render_delay = render_delay

    def _on_step(self) -> bool:
        if self.n_calls % self.visualize_freq == 0 and self.n_calls > 0:
            self._visualize_episode()
        return True

    def _visualize_episode(self) -> None:
        """执行可视化回合，显示规划轨迹"""
        import matplotlib.pyplot as plt

        print(f"\n{'='*50}")
        print(f"[规划型可视化] Step {self.n_calls}")
        print(f"{'='*50}")

        import irsim
        from planning_gym.env import PlanningEnv

        # 创建ir-sim可视化环境
        env_path = os.path.join(os.path.dirname(__file__), "env.yaml")
        viz_env = irsim.make(env_path, display=True, log_level="WARNING")

        # 创建规划环境
        planning_env = PlanningEnv(env_yaml="env.yaml", render_mode=None)
        planning_env._init_env()
        obs, _ = planning_env.reset()

        total_reward = 0
        step_count = 0

        for step in range(self.n_visualize_steps):
            # 获取动作和规划轨迹
            action, _ = self.model.predict(obs, deterministic=True)
            reference_trajectory = planning_env._compute_trajectory_from_action(action)

            # 绘制规划轨迹（绿色虚线）
            traj_points = reference_trajectory[:, :2].T
            viz_env.draw_trajectory(traj_points, traj_type="g--", refresh=True, linewidth=2)

            # 绘制目标点（红色）
            goal_point = planning_env.goal_position.reshape(2, 1)
            viz_env.draw_points(goal_point, s=200, c="red", refresh=True, marker="*")

            # 执行一步
            obs, reward, terminated, truncated, info = planning_env.step(action)
            total_reward += reward
            step_count += 1

            viz_env.render(self.render_delay)

            if terminated or truncated:
                break

        # 结果统计
        result = "✓ 到达目标" if info.get("arrive") else "✗ 碰撞" if info.get("collision") else "○ 超时"
        print(f"结果: {result}")
        print(f"步数: {step_count}")
        print(f"总奖励: {total_reward:.1f}")
        print(f"最后控制: v={info.get('control', [0,0])[0]:.2f}, δ={info.get('control', [0,0])[1]:.2f}")
        print(f"{'='*50}\n")

        viz_env.close()
        planning_env.close()


class DetailedVisualizationCallback(BaseCallback):
    """
    详细可视化回调

    提供更丰富的可视化信息，包括:
    - 车辆轨迹
    - 速度曲线
    - 航向误差
    - 最近障碍物距离
    """

    def __init__(
        self,
        method: str = "control",
        visualize_freq: int = 10000,
        n_visualize_steps: int = 300,
        render_delay: float = 0.02,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.method = method
        self.visualize_freq = visualize_freq
        self.n_visualize_steps = n_visualize_steps
        self.render_delay = render_delay

    def _on_step(self) -> bool:
        if self.n_calls % self.visualize_freq == 0 and self.n_calls > 0:
            self._detailed_visualization()
        return True

    def _detailed_visualization(self) -> None:
        """详细可视化"""
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        print(f"\n{'='*60}")
        print(f"[详细可视化 - {self.method.upper()}] Step {self.n_calls}")
        print(f"{'='*60}")

        # 创建环境和运行回合
        if self.method == "control":
            from control_gym.env import ControlEnv
            env = ControlEnv(env_yaml="env.yaml", render_mode="human")
        else:
            from planning_gym.env import PlanningEnv
            env = PlanningEnv(env_yaml="env.yaml", render_mode="human")

        obs, _ = env.reset()

        # 数据记录
        positions = []
        velocities = []
        heading_errors = []
        rewards = []
        lidar_min_distances = []

        total_reward = 0
        step_count = 0

        fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 3, figure=fig)

        for step in range(self.n_visualize_steps):
            action, _ = self.model.predict(obs, deterministic=True)

            # 记录数据
            robot = env.env.robot
            positions.append([float(robot.state[0, 0]), float(robot.state[1, 0])])
            velocities.append(float(robot.velocity[0, 0]) if robot.velocity is not None else 0)

            # 航向误差
            goal = env.goal_position
            dx = goal[0] - robot.state[0, 0]
            dy = goal[1] - robot.state[1, 0]
            target_heading = np.arctan2(dy, dx)
            heading_err = robot.state[2, 0] - target_heading
            while heading_err > np.pi:
                heading_err -= 2 * np.pi
            while heading_err < -np.pi:
                heading_err += 2 * np.pi
            heading_errors.append(heading_err)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            rewards.append(reward)
            step_count += 1

            # 最近障碍物距离
            lidar_scan = env.env.get_lidar_scan()
            if lidar_scan is not None:
                min_dist = np.min(lidar_scan.get("ranges", [10]))
            else:
                min_dist = 10
            lidar_min_distances.append(min_dist)

            env.render()
            plt.pause(self.render_delay)

            if terminated or truncated:
                break

        # 打印详细统计
        print(f"\n轨迹统计:")
        print(f"  平均速度: {np.mean(velocities):.2f} m/s")
        print(f"  最大速度: {np.max(velocities):.2f} m/s")
        print(f"  平均航向误差: {np.mean(np.abs(heading_errors)) * 180 / np.pi:.1f}°")
        print(f"  最近障碍物: {np.min(lidar_min_distances):.2f} m")

        result = "✓ 到达目标" if info.get("arrive") else "✗ 碰撞" if info.get("collision") else "○ 超时"
        print(f"\n结果: {result}")
        print(f"步数: {step_count}, 总奖励: {total_reward:.1f}")
        print(f"{'='*60}\n")

        env.close()


def create_visualization_callback(
    method: str = "control",
    visualize_freq: int = 5000,
    n_visualize_steps: int = 200,
    render_delay: float = 0.02,
    detailed: bool = False,
):
    """
    创建可视化回调

    Args:
        method: "control" 或 "planning"
        visualize_freq: 可视化频率
        n_visualize_steps: 每次可视化步数
        render_delay: 渲染延迟
        detailed: 是否使用详细可视化

    Returns:
        回调对象
    """
    if detailed:
        return DetailedVisualizationCallback(
            method=method,
            visualize_freq=visualize_freq,
            n_visualize_steps=n_visualize_steps,
            render_delay=render_delay,
        )

    if method == "control":
        return ControlVisualizeCallback(
            visualize_freq=visualize_freq,
            n_visualize_steps=n_visualize_steps,
            render_delay=render_delay,
        )
    elif method == "planning":
        return PlanningVisualizeCallback(
            visualize_freq=visualize_freq,
            n_visualize_steps=n_visualize_steps,
            render_delay=render_delay,
        )
    else:
        raise ValueError(f"Unknown method: {method}")