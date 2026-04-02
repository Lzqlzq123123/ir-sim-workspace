"""
规划型自动驾驶评估脚本

加载训练好的模型进行评估
"""

import os
import argparse
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import torch
from gymnasium import Env
from stable_baselines3 import PPO

from .env import PlanningEnv


class PlanningGymWrapper(Env):
    """Gymnasium环境包装器"""

    def __init__(self, env_yaml: str = "env.yaml", render_mode: Optional[str] = None, **kwargs):
        super().__init__()
        self.env = PlanningEnv(env_yaml=env_yaml, render_mode=render_mode, **kwargs)

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
    save_results: bool = True,
    results_dir: str = "./eval_results",
):
    """评估模型"""
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)

    render_mode = "human" if render else None
    env = PlanningGymWrapper(env_yaml=env_yaml, render_mode=render_mode)

    # 评估统计
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
            results_path / "eval_results.npz",
            **{k: v for k, v in results.items() if isinstance(v, (list, np.ndarray, float, int))}
        )
        print(f"\nResults saved to {results_path}")

    env.close()
    return results


def visualize_policy(
    model_path: str,
    env_yaml: str = "env.yaml",
    max_steps: int = 1000,
    save_animation: bool = False,
    animation_name: str = "planning_policy",
):
    """
    可视化策略执行过程，包括轨迹显示
    """
    import irsim

    model = PPO.load(model_path)

    env_path = Path(__file__).parent.parent / env_yaml
    env = irsim.make(str(env_path), display=True, save_ani=save_animation)

    planning_env = PlanningEnv(env_yaml=env_yaml, render_mode=None)
    planning_env._init_env()

    for step in range(max_steps):
        obs = planning_env._get_observation()

        action, _ = model.predict(obs, deterministic=True)

        # 计算参考轨迹
        reference_trajectory = planning_env._compute_trajectory_from_action(action)

        # 绘制参考轨迹
        traj_points = reference_trajectory[:, :2].T  # [2, N]
        env.draw_trajectory(traj_points, traj_type="g--", refresh=True)

        # 执行一步
        obs, reward, terminated, truncated, info = planning_env.step(action)
        env.render(0.05)

        if planning_env.env.robot.collision_flag:
            print(f"Collision at step {step}")
            break
        if planning_env.env.robot.arrive_flag:
            print(f"Arrived at step {step}")
            break

    env.end(3, ani_name=animation_name)


def main():
    parser = argparse.ArgumentParser(description="Evaluate planning-based autonomous driving model")

    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--env-yaml", type=str, default="env.yaml", help="Environment YAML file")
    parser.add_argument("--n-episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy")
    parser.add_argument("--save-results", action="store_true", help="Save evaluation results")
    parser.add_argument("--results-dir", type=str, default="./eval_results", help="Results directory")
    parser.add_argument("--visualize", action="store_true", help="Visualize policy execution with trajectory")
    parser.add_argument("--save-animation", action="store_true", help="Save animation")
    parser.add_argument("--animation-name", type=str, default="planning_policy", help="Animation name")

    args = parser.parse_args()

    if args.visualize:
        visualize_policy(
            model_path=args.model_path,
            env_yaml=args.env_yaml,
            save_animation=args.save_animation,
            animation_name=args.animation_name,
        )
    else:
        evaluate(
            model_path=args.model_path,
            env_yaml=args.env_yaml,
            n_episodes=args.n_episodes,
            render=not args.no_render,
            deterministic=not args.stochastic,
            save_results=args.save_results,
            results_dir=args.results_dir,
        )


if __name__ == "__main__":
    main()