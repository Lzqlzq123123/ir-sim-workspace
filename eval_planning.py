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

from gymnasium import Env
from stable_baselines3 import PPO

from planning_gym.env import (
    PlanningEnv,
    VehicleKinematics,
    MPCTracker,
    load_env_config,
    resolve_env_yaml_path,
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
    env_config = load_env_config(env_yaml)

    # 从模型路径提取名称
    model_name = Path(model_path).stem  # 获取文件名（不含扩展名）
    if model_name == "final_model":
        # 如果是final_model，使用父文件夹名称
        model_name = Path(model_path).parent.name

    # GIF保存路径：脚本所在目录/gifs/
    script_dir = Path(__file__).parent
    gif_dir = script_dir / "gifs"
    gif_dir.mkdir(exist_ok=True)
    gif_path = str(gif_dir / model_name)  # 不含.gif扩展名

    # 使用irsim.make创建带显示的环境，支持保存GIF
    env_path = resolve_env_yaml_path(env_yaml)
    viz_env = irsim.make(str(env_path), display=True, save_ani=save_gif, log_level="WARNING")

    # 运动学模型参数
    v_max = env_config["v_max"]
    delta_max = env_config["delta_max"]
    a_max = 3.0
    horizon = env_config["planning_horizon"]
    wheelbase = env_config["wheelbase"]
    dt = env_config["step_time"]

    v_min = 0.0 if forward_only else -v_max

    kinematics = VehicleKinematics(
        wheelbase=wheelbase, dt=dt, v_min=v_min, v_max=v_max, delta_max=delta_max, a_max=a_max
    )

    # MPC跟踪器（与训练时一致）
    mpc = MPCTracker(
        horizon=horizon, dt=dt, wheelbase=wheelbase, v_min=v_min, v_max=v_max, delta_max=delta_max
    )

    # 获取目标位置和生成全局路径
    robot = viz_env.robot
    start_position = robot.state[:2].flatten().copy()
    goal_position = robot.goal[:2].flatten() if robot.goal is not None else np.array([50.0, 25.0])

    # 生成全局路径
    num_points = 100
    t = np.linspace(0, 1, num_points)
    global_path = np.zeros((num_points, 4))
    global_path[:, 0] = start_position[0] + t * (goal_position[0] - start_position[0])
    global_path[:, 1] = start_position[1] + t * (goal_position[1] - start_position[1])
    global_path[:, 2] = np.arctan2(goal_position[1] - start_position[1], goal_position[0] - start_position[0])
    global_path[:, 3] = env_config["target_velocity"]

    # 手动构建观测（不使用PlanningEnv，避免matplotlib后端问题）
    lidar_points = env_config["lidar_points"]
    lidar_range_max = env_config["lidar_range_max"]
    target_velocity = env_config["target_velocity"]
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
            [np.clip(delta / max(delta_max, 1e-6), -1, 1)],
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
        control[0] = np.clip(control[0], 0.0 if forward_only else -v_max, v_max)
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
        viz_env.end(ending_time=1, ani_name=gif_path)
        print(f"GIF saved as {gif_path}.gif")
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
