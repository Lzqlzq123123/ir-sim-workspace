#!/usr/bin/env python
"""
规划型自动驾驶训练脚本

支持实时可视化：训练时每一步都显示车辆运动和规划轨迹

使用方法:
  python train_planning.py --total-timesteps 1000000                 # 无可视化训练
  python train_planning.py --total-timesteps 1000000 --visualize     # 实时可视化训练

可视化说明:
  - 车辆运动轨迹
  - 全局参考路径（蓝色虚线）- 从起点到终点的直线
  - 规划轨迹（绿色虚线）- 模型预测的轨迹
  - 目标点（红色星号）
"""

import os
import sys
# Add project root and ir-sim to path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
_ir_sim_path = os.path.join(_project_root, 'ir-sim')
if os.path.exists(_ir_sim_path):
    sys.path.insert(0, _ir_sim_path)

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="规划型自动驾驶PPO训练")
    parser.add_argument("--visualize", action="store_true", help="实时可视化训练过程")
    parser.add_argument("--render-delay", type=float, default=0.01, help="渲染帧间隔(秒)")
    parser.add_argument("--forward-only", action="store_true", default=True, help="只允许车辆前进（默认开启）")
    parser.add_argument("--no-forward-only", action="store_true", help="允许车辆后退")
    parser.add_argument("--env-yaml", type=str, default="env.yaml")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-dir", type=str, default="./logs")
    parser.add_argument("--save-freq", type=int, default=10000)
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--n-eval-episodes", type=int, default=5)
    parser.add_argument("--n-envs", type=int, default=1, help="并行环境数量，>1时使用SubprocVecEnv多进程")
    return parser.parse_args()

args = parse_args()

# 处理forward-only参数
if args.no_forward_only:
    args.forward_only = False

# 关键：在导入任何模块之前设置matplotlib后端
if args.visualize:
    os.environ['MPLBACKEND'] = 'TkAgg'
else:
    os.environ['MPLBACKEND'] = 'Agg'

import matplotlib
matplotlib.use(os.environ.get('MPLBACKEND', 'Agg'))

import numpy as np
from datetime import datetime
from pathlib import Path

try:
    import yaml
except Exception:
    yaml = None

from gymnasium import Env, spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# 导入自定义回调
from callbacks import DetailedMetricsCallback

# 现在才导入irsim，此时后端已经设置好
import irsim


class VisualizedPlanningWrapper(Env):
    """
    带实时可视化的规划型环境包装器

    每一步都会:
    1. 渲染车辆运动
    2. 显示全局参考路径（蓝色虚线）
    3. 显示规划的预测轨迹（绿色虚线）
    4. 显示目标点（红色星号）
    """

    def __init__(self, env_yaml="env.yaml", render_delay=0.01, forward_only=True):
        super().__init__()
        from planning_gym.env import (
            VehicleKinematics,
            MPCTracker,
            load_lidar_config_from_yaml,
            load_obstacle_avoidance_config_from_yaml,
            load_robot_shape_config_from_yaml,
        )

        self.render_delay = render_delay
        self.env_yaml = env_yaml
        self.forward_only = forward_only

        # 创建ir-sim可视化环境
        env_path = os.path.join(os.path.dirname(__file__), env_yaml)
        self.viz_env = irsim.make(env_path, display=True, log_level="WARNING")

        # 从env.yaml读取配置（与env.yaml保持一致）
        self.target_velocity = 3.0  # 与env.yaml中的vel_max一致
        self.max_steps = 1000
        lidar_cfg = load_lidar_config_from_yaml(env_yaml)
        self.lidar_points = int(lidar_cfg.get("number", 100))
        self.lidar_range_max = float(lidar_cfg.get("range_max", 10.0))
        obstacle_cfg = load_obstacle_avoidance_config_from_yaml(env_yaml)
        self.obstacle_min_distance = float(obstacle_cfg.get("min_distance", 3.0))
        self.obstacle_danger_distance = float(obstacle_cfg.get("danger_distance", 1.0))
        self.horizon = 10
        shape_cfg = load_robot_shape_config_from_yaml(env_yaml)
        self.wheelbase = float(shape_cfg.get("wheelbase", 1.75))
        self.v_max = 3.0  # 与env.yaml中的vel_max一致
        self.delta_max = 1.0  # 与env.yaml中的vel_max[1]一致
        self._global_path_cfg = self._load_global_path_config(env_yaml)
        random_cfg = self._load_randomization_config(env_yaml)
        self.randomize_every_reset = bool(random_cfg.get("enabled", False))
        self.min_start_goal_distance = float(random_cfg.get("min_start_goal_distance", 10.0))
        self.random_margin = float(random_cfg.get("random_margin", 1.5))
        self.randomize_obstacles = bool(random_cfg.get("randomize_obstacles", True))
        self.max_randomization_attempts = int(random_cfg.get("max_sampling_attempts", 200))
        self._random_obstacle_ids = [
            obs.id for obs in self.viz_env.obstacle_list if getattr(obs, "shape", "") == "circle"
        ]

        # 运动学模型和MPC跟踪器
        # 根据forward_only设置速度范围
        v_min = 0.0 if forward_only else -self.v_max
        self.kinematics = VehicleKinematics(
            wheelbase=self.wheelbase, dt=0.1, v_min=v_min, v_max=self.v_max, delta_max=self.delta_max
        )
        self.mpc = MPCTracker(
            horizon=self.horizon, dt=0.1, wheelbase=self.wheelbase, v_min=v_min, v_max=self.v_max, delta_max=self.delta_max
        )

        # 状态
        self.goal_position = None
        self.start_position = None
        self.global_path = None
        self.local_path = None  # 局部路径
        self.local_path_length = 20  # 局部轨迹点数
        self._closest_idx = 0  # 当前最近点索引
        self.current_step = 0
        self.prev_control = np.array([self.v_max / 2, 0.0])  # 初始控制量
        self.prev_distance = None

        # 初始化时先对齐到CSV首尾，避免占位state=[0,0,...]在可视化中留下残影
        self._bootstrap_robot_pose_from_csv_once()

        # 空间定义
        self.state_dim = 4 + self.lidar_points
        self.action_dim = 2  # [acceleration, steering]
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.state_dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (self.action_dim,), np.float32)

    def _bootstrap_robot_pose_from_csv_once(self):
        """环境创建后立即将机器人/目标对齐到CSV首尾，避免(0,0)占位状态被绘制。"""
        csv_path = self._load_global_path_from_csv()
        if csv_path is None or len(csv_path) < 2:
            return

        self.global_path = csv_path
        self.start_position = self.global_path[0, :2].copy()
        self.goal_position = self.global_path[-1, :2].copy()

        robot = self.viz_env.robot
        start = self.global_path[0]
        end = self.global_path[-1]

        aligned_state = robot.state.copy()
        aligned_state[0, 0] = float(start[0])
        aligned_state[1, 0] = float(start[1])
        aligned_state[2, 0] = float(start[2])
        if aligned_state.shape[0] > 3:
            aligned_state[3, 0] = 0.0

        robot.set_state(aligned_state, init=True)
        if robot.velocity is not None:
            robot.set_velocity(np.zeros_like(robot.velocity), init=True)
        robot.set_goal([float(end[0]), float(end[1]), float(end[2])], init=True)
        self.viz_env.build_tree()

    def _load_randomization_config(self, env_yaml: str):
        if yaml is None:
            return {}

        env_path = Path(os.path.join(os.path.dirname(__file__), env_yaml))
        if not env_path.exists():
            return {}

        try:
            with open(env_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            gui_cfg = data.get("gui", {}) if isinstance(data, dict) else {}
            cfg = gui_cfg.get("randomization", {}) if isinstance(gui_cfg, dict) else {}
            return cfg if isinstance(cfg, dict) else {}
        except Exception:
            return {}

    def _load_global_path_config(self, env_yaml: str):
        if yaml is None:
            return {}

        env_path = Path(os.path.join(os.path.dirname(__file__), env_yaml))
        if not env_path.exists():
            return {}

        try:
            with open(env_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            gui_cfg = data.get("gui", {}) if isinstance(data, dict) else {}
            cfg = gui_cfg.get("global_path", {}) if isinstance(gui_cfg, dict) else {}
            return cfg if isinstance(cfg, dict) else {}
        except Exception:
            return {}

    def _load_global_path_from_csv(self):
        cfg = self._global_path_cfg or {}
        csv_path = cfg.get("csv_path", "")
        if not csv_path:
            return None

        csv_file = Path(str(csv_path))
        if not csv_file.is_absolute():
            csv_file = (Path(os.path.dirname(__file__)) / csv_file).resolve()

        if not csv_file.exists():
            return None

        delimiter = str(cfg.get("delimiter", ","))
        x_col = str(cfg.get("x_col", "x"))
        y_col = str(cfg.get("y_col", "y"))
        yaw_col = str(cfg.get("yaw_col", "yaw"))
        speed_col = str(cfg.get("speed_col", "speed"))

        try:
            data = np.genfromtxt(
                str(csv_file), delimiter=delimiter, names=True, dtype=None, encoding="utf-8"
            )
        except Exception:
            return None

        if data is None or getattr(data, "dtype", None) is None or data.dtype.names is None:
            return None

        if data.ndim == 0:
            data = np.array([data], dtype=data.dtype)

        names = set(data.dtype.names)
        if x_col not in names or y_col not in names:
            return None

        x = np.asarray(data[x_col], dtype=float)
        y = np.asarray(data[y_col], dtype=float)
        if x.size < 2:
            return None

        if yaw_col in names:
            yaw = np.asarray(data[yaw_col], dtype=float)
        else:
            dx = np.gradient(x)
            dy = np.gradient(y)
            yaw = np.arctan2(dy, dx)

        if speed_col in names:
            v = np.asarray(data[speed_col], dtype=float)
        else:
            v = np.ones_like(x) * self.target_velocity

        n = min(x.size, y.size, yaw.size, v.size)
        path = np.zeros((n, 4), dtype=float)
        path[:, 0] = x[:n]
        path[:, 1] = y[:n]
        path[:, 2] = yaw[:n]
        path[:, 3] = v[:n]
        return path

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        if self.randomize_every_reset:
            self._randomize_episode_layout()

        # 重置ir-sim环境
        self.viz_env.reset()

        robot = self.viz_env.robot

        # 获取起点
        self.start_position = robot.state[:2].flatten().copy()

        # 获取目标点
        if robot.goal is not None:
            self.goal_position = robot.goal[:2].flatten()
        else:
            self.goal_position = np.array([50.0, 25.0])

        # 生成全局路径
        self._generate_global_path()
        self._apply_csv_start_goal_if_needed()

        self.current_step = 0
        self.prev_control = np.array([self.v_max / 2, 0.0])  # 初始控制量
        self.prev_distance = None

        # 初始渲染
        self._draw_global_path()
        self._draw_goal()
        self.viz_env.render(self.render_delay)

        obs = self._get_observation()
        return obs, {}

    def _is_collision_free_state(self, robot, candidate_state):
        original_state = robot.state.copy()
        robot.set_state(candidate_state, init=False)
        collision = any(robot.check_collision(obj) for obj in self.viz_env.obstacle_list)
        robot.set_state(original_state, init=False)
        return not collision

    def _sample_free_robot_state(self, robot, width, height):
        x_low, x_high = self.random_margin, max(self.random_margin + 0.1, width - self.random_margin)
        y_low, y_high = self.random_margin, max(self.random_margin + 0.1, height - self.random_margin)

        for _ in range(self.max_randomization_attempts):
            candidate = robot.state.copy()
            candidate[0, 0] = np.random.uniform(x_low, x_high)
            candidate[1, 0] = np.random.uniform(y_low, y_high)
            candidate[2, 0] = np.random.uniform(-np.pi, np.pi)
            if candidate.shape[0] > 3:
                candidate[3, 0] = 0.0

            if self._is_collision_free_state(robot, candidate):
                return candidate

        return None

    def _sample_free_goal(self, robot, width, height):
        x_low, x_high = self.random_margin, max(self.random_margin + 0.1, width - self.random_margin)
        y_low, y_high = self.random_margin, max(self.random_margin + 0.1, height - self.random_margin)

        for _ in range(self.max_randomization_attempts):
            candidate = robot.state.copy()
            candidate[0, 0] = np.random.uniform(x_low, x_high)
            candidate[1, 0] = np.random.uniform(y_low, y_high)
            candidate[2, 0] = np.random.uniform(-np.pi, np.pi)
            if candidate.shape[0] > 3:
                candidate[3, 0] = 0.0

            if self._is_collision_free_state(robot, candidate):
                return np.array([candidate[0, 0], candidate[1, 0], candidate[2, 0]], dtype=float)

        return None

    def _randomize_episode_layout(self):
        robot = self.viz_env.robot
        width = float(getattr(self.viz_env.world_param, "width", 37.0))
        height = float(getattr(self.viz_env.world_param, "height", 20.0))

        if self.randomize_obstacles and len(self._random_obstacle_ids) > 0:
            low = [self.random_margin, self.random_margin, -np.pi]
            high = [max(self.random_margin + 0.1, width - self.random_margin),
                    max(self.random_margin + 0.1, height - self.random_margin),
                    np.pi]
            self.viz_env.random_obstacle_position(low, high, ids=self._random_obstacle_ids, non_overlapping=True)

        selected_start = None
        selected_goal = None
        for _ in range(self.max_randomization_attempts):
            start_state = self._sample_free_robot_state(robot, width, height)
            goal = self._sample_free_goal(robot, width, height)
            if start_state is None or goal is None:
                continue

            if np.linalg.norm(start_state[:2, 0] - goal[:2]) >= self.min_start_goal_distance:
                selected_start = start_state
                selected_goal = goal
                break

        if selected_start is None or selected_goal is None:
            return

        robot.set_state(selected_start, init=True)
        if robot.velocity is not None:
            robot.set_velocity(np.zeros_like(robot.velocity), init=True)
        robot.set_goal(selected_goal.tolist(), init=True)
        self.viz_env.build_tree()

    def step(self, action):
        # 将归一化动作转换为实际值
        a_max = 3.0
        acceleration = action[0] * a_max
        steering = action[1] * self.delta_max

        # 获取当前状态
        robot = self.viz_env.robot
        v = float(robot.velocity[0, 0]) if robot.velocity is not None else 0.0

        # 确保当前速度非负
        if self.forward_only:
            v = max(v, 0.0)

        current_state = np.array([
            float(robot.state[0, 0]),
            float(robot.state[1, 0]),
            float(robot.state[2, 0]),
            v,
        ])

        # 运动学模型预测轨迹
        reference_trajectory = self.kinematics.predict_trajectory(
            current_state, acceleration, steering, self.horizon
        )

        # MPC跟踪轨迹，输出控制量
        control = self.mpc.compute_control(current_state, reference_trajectory)

        # 限制控制量范围
        control[0] = np.clip(control[0], 0.0 if self.forward_only else -self.v_max, self.v_max)
        control[1] = np.clip(control[1], -self.delta_max, self.delta_max)

        self.prev_control = control

        # 绘制全局路径、规划轨迹和目标点
        self._draw_global_path()
        self._draw_planned_trajectory(reference_trajectory)
        self._draw_goal()

        # 执行控制
        self.viz_env.step(control)
        self.viz_env.render(self.render_delay)

        self.current_step += 1

        # 获取观测
        obs = self._get_observation()

        # 检查终止条件
        terminated = False
        truncated = False
        if robot.collision_flag:
            terminated = True
        elif robot.arrive_flag:
            terminated = True
        elif self.current_step >= self.max_steps:
            truncated = True

        info = {
            "collision": robot.collision_flag,
            "arrive": robot.arrive_flag,
            "step": self.current_step,
        }

        reward = self._compute_reward(info)
        return obs, reward, terminated, truncated, info

    def _generate_global_path(self):
        """生成全局参考路径（仅支持CSV轨迹）"""
        csv_path = self._load_global_path_from_csv()
        if csv_path is None or len(csv_path) < 2:
            raise ValueError(
                "global_path CSV 无效。请在 env.yaml 的 gui.global_path.csv_path 指定有效CSV，且至少包含2个点。"
            )

        self.global_path = csv_path
        self.start_position = self.global_path[0, :2].copy()
        self.goal_position = self.global_path[-1, :2].copy()

    def _apply_csv_start_goal_if_needed(self):
        """将车辆状态和目标点对齐到CSV轨迹首尾。"""
        if self.global_path is None or len(self.global_path) < 2:
            return

        robot = self.viz_env.robot
        start = self.global_path[0]
        end = self.global_path[-1]

        aligned_state = robot.state.copy()
        aligned_state[0, 0] = float(start[0])
        aligned_state[1, 0] = float(start[1])
        aligned_state[2, 0] = float(start[2])
        if aligned_state.shape[0] > 3:
            aligned_state[3, 0] = 0.0

        robot.set_state(aligned_state, init=True)
        if robot.velocity is not None:
            robot.set_velocity(np.zeros_like(robot.velocity), init=True)
        robot.set_goal([float(end[0]), float(end[1]), float(end[2])], init=True)
        self.viz_env.build_tree()

    def _draw_global_path(self):
        """绘制全局参考路径（蓝色虚线）和局部路径（红色实线）"""
        if self.global_path is not None:
            path_points = self.global_path[:, :2].T
            self.viz_env.draw_trajectory(path_points, traj_type="b--", refresh=True, linewidth=2, label="Global Path")

        # 绘制局部路径（红色实线）
        if self.local_path is not None and len(self.local_path) > 0:
            local_points = self.local_path[:, :2].T
            self.viz_env.draw_trajectory(local_points, traj_type="r-", refresh=False, linewidth=3)

    def _draw_planned_trajectory(self, trajectory):
        """绘制规划轨迹（绿色虚线）"""
        if trajectory is not None:
            traj_points = trajectory[:, :2].T
            self.viz_env.draw_trajectory(traj_points, traj_type="g--", refresh=True, linewidth=2)

    def _draw_goal(self):
        """绘制目标点（红色星号）"""
        if self.goal_position is not None:
            goal_point = self.goal_position.reshape(2, 1)
            self.viz_env.draw_points(goal_point, s=200, c="red", refresh=True, marker="*")

    def _get_observation(self):
        """获取观测"""
        robot = self.viz_env.robot
        robot_state = robot.state.flatten()

        # 速度
        v = float(robot.velocity[0, 0]) if robot.velocity is not None else 0.0

        # 找到全局路径上最近的点
        distances = np.sqrt(
            (self.global_path[:, 0] - robot_state[0])**2 +
            (self.global_path[:, 1] - robot_state[1])**2
        )
        self._closest_idx = np.argmin(distances)

        # 更新局部轨迹：从当前最近点向前切分
        end_idx = min(self._closest_idx + self.local_path_length, len(self.global_path))
        self.local_path = self.global_path[self._closest_idx:end_idx].copy()

        # 获取最近参考点
        ref_point = self.global_path[self._closest_idx]

        # 航向角偏差：基于局部轨迹计算
        delta_theta = self._normalize_angle(robot_state[2] - ref_point[2])

        # 横向偏差：基于局部轨迹计算
        dx = robot_state[0] - ref_point[0]
        dy = robot_state[1] - ref_point[1]
        ref_heading = ref_point[2]
        delta_y = -dx * np.sin(ref_heading) + dy * np.cos(ref_heading)

        # 前轮转角
        delta = robot_state[3] if len(robot_state) > 3 else 0.0

        # 激光雷达
        lidar_scan = self.viz_env.get_lidar_scan()
        if lidar_scan is not None:
            ranges = lidar_scan.get("ranges", np.ones(self.lidar_points) * self.lidar_range_max)
            lidar_normalized = np.clip(ranges / self.lidar_range_max, 0, 1)
        else:
            lidar_normalized = np.ones(self.lidar_points)

        obs = np.concatenate([
            [v / self.target_velocity],
            [delta_theta / np.pi],
            [np.clip(delta_y / 5.0, -1, 1)],
            [np.clip(delta, -1, 1)],
            lidar_normalized
        ])
        return obs.astype(np.float32)

    def _compute_reward(self, info):
        """计算奖励"""
        robot = self.viz_env.robot
        robot_state = robot.state.flatten()

        distance_to_goal = np.sqrt(
            (robot_state[0] - self.goal_position[0])**2 +
            (robot_state[1] - self.goal_position[1])**2
        )

        reward = 0.0

        # === 1. 进度奖励（核心）===
        if self.prev_distance is not None:
            progress = self.prev_distance - distance_to_goal
            reward += progress * 5.0

        self.prev_distance = distance_to_goal

        # === 2. 速度奖励 ===
        v = float(robot.velocity[0, 0]) if robot.velocity is not None else 0.0

        if v > 1.0:
            reward += 0.5
        elif v > 0.5:
            reward += 0.2
        elif v < 0.1:
            reward -= 1.0

        # === 3. 障碍物避障（关键！）===
        lidar_scan = self.viz_env.get_lidar_scan()
        if lidar_scan is not None:
            ranges = lidar_scan.get("ranges", np.ones(self.lidar_points) * self.lidar_range_max)
            min_distance = np.min(ranges)
            if min_distance < self.obstacle_min_distance:
                obstacle_penalty = (self.obstacle_min_distance - min_distance) * 1.0
                reward -= obstacle_penalty
            if min_distance < self.obstacle_danger_distance:
                reward -= 2.0

        # === 4. 目标方向引导（不强制路径）===
        if distance_to_goal > 5.0:
            dx = self.goal_position[0] - robot_state[0]
            dy = self.goal_position[1] - robot_state[1]
            goal_heading = np.arctan2(dy, dx)
            heading_to_goal = abs(self._normalize_angle(goal_heading - robot_state[2]))
            if heading_to_goal > np.pi / 2:
                reward -= 0.1

        # === 5. 碰撞/到达 ===
        if robot.collision_flag:
            reward -= 100.0
        if robot.arrive_flag:
            reward += 100.0

        return reward

    def _normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def close(self):
        self.viz_env.close()


class PlanningGymWrapper(Env):
    """无可视化的环境包装器"""

    def __init__(self, env_yaml="env.yaml", forward_only=True):
        super().__init__()
        from planning_gym.env import PlanningEnv
        self.env = PlanningEnv(
            env_yaml=env_yaml,
            render_mode=None,
        )
        self.forward_only = forward_only
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.env.state_dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (self.env.action_dim,), np.float32)

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        # 如果只允许前进，修改加速度
        if self.forward_only:
            action = action.copy()
            # 限制加速度不能太负（不能急刹车导致倒车）
            if action[0] < -0.3:
                action[0] = -0.3
        return self.env.step(action)

    def close(self):
        self.env.close()


def main():
    if args.seed is not None:
        np.random.seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) / f"planning_ppo_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = save_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    if args.visualize:
        print("\n实时可视化模式: 训练过程中每一步都会显示车辆运动和规划轨迹")
        print("  - 蓝色虚线: 全局参考路径（从起点到终点）")
        print("  - 绿色虚线: 模型预测的规划轨迹")
        print("  - 红色星号: 目标位置")
        if args.forward_only:
            print("  - 只允许前进: 速度不能为负（默认开启）")
        print(f"  渲染帧间隔: {args.render_delay}秒\n")
        env = DummyVecEnv([lambda: Monitor(VisualizedPlanningWrapper(
            env_yaml=args.env_yaml,
            render_delay=args.render_delay,
            forward_only=args.forward_only
        ))])
    else:
        def make_env():
            return Monitor(PlanningGymWrapper(
                env_yaml=args.env_yaml,
                forward_only=args.forward_only
            ))
        if args.n_envs > 1:
            env = SubprocVecEnv([make_env for _ in range(args.n_envs)])
        else:
            env = DummyVecEnv([make_env])

    logger = configure(str(log_dir), ["stdout", "tensorboard"])

    model = PPO(
        "MlpPolicy", env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        verbose=1,
        tensorboard_log=str(log_dir),
        seed=args.seed,
        device=args.device,
    )
    model.set_logger(logger)

    callbacks = [
        CheckpointCallback(args.save_freq, str(save_dir / "checkpoints"), "planning_ppo"),
        DetailedMetricsCallback(),  # 添加成功率、碰撞率跟踪
    ]

    # 不使用EvalCallback，避免环境创建带来的问题
    # 训练完成后可单独运行评估脚本
    # if args.eval_freq > 0 and not args.visualize:
    #     callbacks.append(EvalCallback(eval_env, best_model_save_path=str(save_dir / "best_model"),
    #                         log_path=str(save_dir / "eval_logs"), eval_freq=args.eval_freq,
    #                         n_eval_episodes=args.n_eval_episodes, deterministic=True))

    print(f"\n{'='*60}\n规划型自动驾驶训练\n{'='*60}")
    print(f"开始时间: {timestamp}")
    print(f"保存目录: {save_dir}")
    print(f"总步数: {args.total_timesteps}")
    print(f"实时可视化: {'启用' if args.visualize else '禁用'}")
    print(f"并行环境数: {args.n_envs}")
    print(f"只允许前进: {'启用' if args.forward_only else '禁用'}")
    print(f"动作空间: [加速度, 转角] (归一化到[-1,1])")
    print(f"{'='*60}\n")

    try:
        model.learn(total_timesteps=args.total_timesteps, callback=CallbackList(callbacks), progress_bar=True)
    except KeyboardInterrupt:
        print("\n训练被用户中断")

    model.save(str(save_dir / "final_model"))
    print(f"\n最终模型已保存: {save_dir / 'final_model'}")
    env.close()


if __name__ == "__main__":
    main()
