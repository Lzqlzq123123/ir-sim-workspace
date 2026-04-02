#!/usr/bin/env python
"""
规划型自动驾驶评估脚本

使用方法:
  python eval_planning.py --model-path ./logs/planning_ppo_xxx/final_model.zip
  python eval_planning.py --model-path ./logs/planning_ppo_xxx/final_model.zip --visualize
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
    parser = argparse.ArgumentParser(description="Evaluate planning-based autonomous driving model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--env-yaml", type=str, default="env.yaml", help="Environment YAML file")
    parser.add_argument("--n-episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy")
    parser.add_argument("--save-results", action="store_true", help="Save evaluation results")
    parser.add_argument("--results-dir", type=str, default="./eval_results", help="Results directory")
    parser.add_argument("--visualize", action="store_true", help="Visualize policy execution")
    parser.add_argument("--no-save-gif", action="store_true", help="Disable GIF saving (default: save GIF)")
    parser.add_argument("--render-delay", type=float, default=0.05, help="Render delay in seconds")
    parser.add_argument("--no-forward-only", action="store_true", help="Allow backward motion")
    return parser.parse_args()

args = parse_args()

# 根据可视化参数设置matplotlib后端（必须在导入matplotlib/irsim之前）
if args.visualize:
    os.environ['MPLBACKEND'] = 'TkAgg'
else:
    os.environ['MPLBACKEND'] = 'Agg'

import matplotlib
matplotlib.use(os.environ['MPLBACKEND'])

import numpy as np
from pathlib import Path
from typing import Optional

try:
    import yaml
except Exception:
    yaml = None

from gymnasium import Env
from stable_baselines3 import PPO

from planning_gym.env import (
    PlanningEnv,
    VehicleKinematics,
    MPCTracker,
    load_lidar_config_from_yaml,
    load_robot_shape_config_from_yaml,
)
import irsim


class PlanningGymWrapper(Env):
    """Gymnasium环境包装器"""

    def __init__(self, env_yaml: str = "env.yaml", render_mode: Optional[str] = None, forward_only: bool = True, **kwargs):
        super().__init__()
        self.env = PlanningEnv(env_yaml=env_yaml, render_mode=render_mode, forward_only=forward_only, **kwargs)

        from gymnasium import spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.env.state_dim,),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.env.action_dim,),
            dtype=np.float32
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


def evaluate(
    model_path: str,
    env_yaml: str = "env.yaml",
    n_episodes: int = 10,
    render: bool = True,
    deterministic: bool = True,
    forward_only: bool = True,
    save_results: bool = True,
    results_dir: str = "./eval_results",
):
    """评估模型"""
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)

    render_mode = "human" if render else None
    env = PlanningGymWrapper(env_yaml=env_yaml, render_mode=render_mode, forward_only=forward_only)

    episode_rewards = []
    episode_lengths = []
    episode_collisions = 0
    episode_arrivals = 0
    episode_timeouts = 0

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            step_count += 1

            if render:
                env.render()

        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)

        if info.get("collision", False):
            episode_collisions += 1
        elif info.get("arrive", False):
            episode_arrivals += 1
        else:
            episode_timeouts += 1

        print(f"Episode {episode + 1}/{n_episodes}: "
              f"Reward={total_reward:.2f}, Steps={step_count}, "
              f"Result={'Arrived' if info.get('arrive') else 'Collision' if info.get('collision') else 'Timeout'}")

    results = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "collision_rate": episode_collisions / n_episodes,
        "arrival_rate": episode_arrivals / n_episodes,
        "timeout_rate": episode_timeouts / n_episodes,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
    }

    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Mean Reward: {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
    print(f"Mean Length: {results['mean_length']:.1f} +/- {results['std_length']:.1f}")
    print(f"Arrival Rate: {results['arrival_rate'] * 100:.1f}%")
    print(f"Collision Rate: {results['collision_rate'] * 100:.1f}%")
    print(f"Timeout Rate: {results['timeout_rate'] * 100:.1f}%")

    if save_results:
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        np.savez(
            results_path / "planning_eval_results.npz",
            **{k: v for k, v in results.items() if isinstance(v, (list, np.ndarray, float, int))}
        )
        print(f"\nResults saved to {results_path}")

    env.close()
    return results


def visualize_policy(
    model_path: str,
    env_yaml: str = "env.yaml",
    max_steps: int = 1000,
    forward_only: bool = True,
    save_gif: bool = True,
    render_delay: float = 0.05,
):
    """可视化策略执行过程，包括轨迹显示

    可视化说明:
    - 蓝色虚线: 全局路径（从起点到终点）
    - 红色实线: 局部路径（当前参考轨迹段，用于计算偏差）
    - 绿色虚线: 规划轨迹（模型预测的未来轨迹）
    """
    model = PPO.load(model_path)

    # 从模型路径提取名称
    model_name = Path(model_path).stem  # 获取文件名（不含扩展名）
    if model_name == "final_model":
        # 如果是final_model，使用父文件夹名称
        model_name = Path(model_path).parent.name

    # ir-sim会将ani_name拼接到自身animation目录下，这里只传文件名，避免绝对路径报错
    gif_name = model_name

    # 使用irsim.make创建带显示的环境，支持保存GIF
    env_path = Path(__file__).parent / env_yaml
    viz_env = irsim.make(str(env_path), display=True, save_ani=save_gif, log_level="WARNING")

    # 运动学模型参数
    v_max = 3.0
    delta_max = 1.0
    a_max = 3.0
    horizon = 10
    shape_cfg = load_robot_shape_config_from_yaml(env_yaml)
    wheelbase = float(shape_cfg.get("wheelbase", 1.75))

    kinematics = VehicleKinematics(
        wheelbase=wheelbase, dt=0.1, v_min=0.0, v_max=v_max, delta_max=delta_max
    )

    # MPC跟踪器（与训练时一致）
    mpc = MPCTracker(
        horizon=horizon, dt=0.1, wheelbase=wheelbase, v_min=0.0, v_max=v_max, delta_max=delta_max
    )

    # 从env.yaml读取CSV全局路径并对齐机器人起终点（与训练保持一致）
    def _load_global_path_config(env_yaml_path: Path):
        if yaml is None or not env_yaml_path.exists():
            return {}
        try:
            with open(env_yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            gui_cfg = data.get("gui", {}) if isinstance(data, dict) else {}
            cfg = gui_cfg.get("global_path", {}) if isinstance(gui_cfg, dict) else {}
            return cfg if isinstance(cfg, dict) else {}
        except Exception:
            return {}

    def _load_global_path_from_csv(cfg: dict, env_yaml_path: Path):
        csv_path = cfg.get("csv_path", "")
        if not csv_path:
            return None

        csv_file = Path(str(csv_path))
        if not csv_file.is_absolute():
            csv_file = (env_yaml_path.parent / csv_file).resolve()
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
            v = np.ones_like(x) * 3.0

        n = min(x.size, y.size, yaw.size, v.size)
        path = np.zeros((n, 4), dtype=float)
        path[:, 0] = x[:n]
        path[:, 1] = y[:n]
        path[:, 2] = yaw[:n]
        path[:, 3] = v[:n]
        return path

    gp_cfg = _load_global_path_config(env_path)
    global_path = _load_global_path_from_csv(gp_cfg, env_path)
    if global_path is None or len(global_path) < 2:
        raise ValueError("global_path CSV 无效，请检查 env.yaml 的 gui.global_path.csv_path 和列名配置")

    robot = viz_env.robot
    start = global_path[0]
    end = global_path[-1]
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
    viz_env.build_tree()

    # 手动构建观测（不使用PlanningEnv，避免matplotlib后端问题）
    lidar_cfg = load_lidar_config_from_yaml(env_yaml)
    lidar_points = int(lidar_cfg.get("number", 100))
    lidar_range_max = float(lidar_cfg.get("range_max", 10.0))
    target_velocity = 3.0
    local_path_length = 20

    def get_observation():
        robot_state = viz_env.robot.state.flatten()
        v = 0.0
        if viz_env.robot.velocity is not None and len(viz_env.robot.velocity) > 0:
            v = float(viz_env.robot.velocity.flatten()[0])

        # 找最近点
        distances = np.sqrt((global_path[:, 0] - robot_state[0])**2 + (global_path[:, 1] - robot_state[1])**2)
        closest_idx = np.argmin(distances)
        ref_point = global_path[closest_idx]

        # 航向角偏差
        delta_theta = robot_state[2] - ref_point[2]
        while delta_theta > np.pi:
            delta_theta -= 2 * np.pi
        while delta_theta < -np.pi:
            delta_theta += 2 * np.pi

        # 横向偏差
        dx = robot_state[0] - ref_point[0]
        dy = robot_state[1] - ref_point[1]
        ref_heading = ref_point[2]
        delta_y = -dx * np.sin(ref_heading) + dy * np.cos(ref_heading)

        # 前轮转角
        delta = robot_state[3] if len(robot_state) > 3 else 0.0

        # 激光雷达
        lidar_scan = viz_env.get_lidar_scan()
        if lidar_scan is not None:
            ranges = lidar_scan.get("ranges", np.ones(lidar_points) * lidar_range_max)
            lidar_normalized = np.clip(ranges / lidar_range_max, 0, 1)
        else:
            lidar_normalized = np.ones(lidar_points)

        obs = np.concatenate([
            [v / max(target_velocity, 0.1)],
            [delta_theta / np.pi],
            [np.clip(delta_y / 5.0, -1, 1)],
            [np.clip(delta / 1.0, -1, 1)],
            lidar_normalized
        ])
        return obs.astype(np.float32), closest_idx

    obs, _ = get_observation()

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)

        # 将归一化动作转换为实际值
        acceleration = action[0] * a_max
        steering = action[1] * delta_max

        # 获取当前状态用于轨迹预测
        robot = viz_env.robot
        v = float(robot.velocity[0, 0]) if robot.velocity is not None else 0.0
        if forward_only:
            v = max(v, 0.0)

        current_state = np.array([
            float(robot.state[0, 0]),
            float(robot.state[1, 0]),
            float(robot.state[2, 0]),
            v,
        ])

        # 使用运动学模型预测轨迹
        planned_trajectory = kinematics.predict_trajectory(
            current_state, acceleration, steering, horizon
        )

        # 绘制全局路径
        path_points = global_path[:, :2].T
        viz_env.draw_trajectory(path_points, traj_type="b--", refresh=True, linewidth=2)

        # 绘制局部路径
        robot_state = viz_env.robot.state.flatten()
        distances = np.sqrt((global_path[:, 0] - robot_state[0])**2 + (global_path[:, 1] - robot_state[1])**2)
        closest_idx = np.argmin(distances)
        end_idx = min(closest_idx + local_path_length, len(global_path))
        local_path = global_path[closest_idx:end_idx]
        if len(local_path) > 0:
            local_points = local_path[:, :2].T
            viz_env.draw_trajectory(local_points, traj_type="r-", refresh=False, linewidth=3)

        # 绘制规划轨迹
        traj_points = planned_trajectory[:, :2].T
        viz_env.draw_trajectory(traj_points, traj_type="g--", refresh=False, linewidth=2)

        # 使用MPC跟踪轨迹（与训练时一致）
        control = mpc.compute_control(current_state, planned_trajectory)
        control[0] = np.clip(control[0], 0.0, v_max)
        control[1] = np.clip(control[1], -delta_max, delta_max)

        viz_env.step(control)
        viz_env.render(render_delay)

        # 获取新观测
        obs, _ = get_observation()

        # 检查终止条件
        robot = viz_env.robot
        if robot.collision_flag:
            print(f"Collision at step {step}")
            break
        elif robot.arrive_flag:
            print(f"Arrived at step {step}")
            break

    # 保存GIF
    if save_gif:
        viz_env.end(ending_time=1, ani_name=gif_name)
        print(f"GIF saved as {gif_name}.gif (in ir-sim/animation)")
    else:
        viz_env.end(ending_time=1)


def main():
    forward_only = not args.no_forward_only

    if args.visualize:
        visualize_policy(
            model_path=args.model_path,
            env_yaml=args.env_yaml,
            forward_only=forward_only,
            save_gif=not args.no_save_gif,
            render_delay=args.render_delay,
        )
    else:
        evaluate(
            model_path=args.model_path,
            env_yaml=args.env_yaml,
            n_episodes=args.n_episodes,
            render=not args.no_render,
            deterministic=not args.stochastic,
            forward_only=forward_only,
            save_results=args.save_results,
            results_dir=args.results_dir,
        )


if __name__ == "__main__":
    main()
