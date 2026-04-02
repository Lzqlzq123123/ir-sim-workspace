"""
规划型自动驾驶TD3训练脚本
"""

import os
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from gymnasium import Env
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

try:
    from .env import PlanningEnv
except ImportError:  # 兼容直接执行 python planning_gym/train.py
    from env import PlanningEnv


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


def make_env(env_yaml: str, render_mode: Optional[str] = None, rank: int = 0):
    """创建环境的工厂函数"""
    def _init():
        env = PlanningGymWrapper(env_yaml=env_yaml, render_mode=render_mode)
        env = Monitor(env)
        return env
    return _init


def train(args):
    """训练主函数"""
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) / f"planning_td3_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    log_dir = save_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    # 创建环境
    if args.n_envs > 1:
        env = SubprocVecEnv([
            make_env(args.env_yaml, render_mode=None, rank=i)
            for i in range(args.n_envs)
        ])
    else:
        env = DummyVecEnv([make_env(args.env_yaml, render_mode=None)])

    # 创建评估环境
    eval_env = DummyVecEnv([make_env(args.env_yaml, render_mode=None)])

    # 配置日志
    logger = configure(str(log_dir), ["stdout", "tensorboard"])

    sample_env = env.envs[0] if hasattr(env, "envs") else env
    action_dim = int(np.prod(sample_env.action_space.shape))
    action_noise = NormalActionNoise(
        mean=np.zeros(action_dim, dtype=np.float32),
        sigma=np.ones(action_dim, dtype=np.float32) * args.action_noise_std,
    )

    # 创建TD3模型
    model = TD3(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        train_freq=(args.train_freq, "step"),
        gradient_steps=args.gradient_steps,
        gamma=args.gamma,
        tau=args.tau,
        policy_delay=args.policy_delay,
        target_policy_noise=args.target_policy_noise,
        target_noise_clip=args.target_noise_clip,
        action_noise=action_noise,
        verbose=1,
        tensorboard_log=str(log_dir),
        seed=args.seed,
        device=args.device,
    )

    model.set_logger(logger)

    # 创建回调
    callbacks = []

    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(save_dir / "checkpoints"),
        name_prefix="planning_td3",
    )
    callbacks.append(checkpoint_callback)

    if args.eval_freq > 0:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(save_dir / "best_model"),
            log_path=str(save_dir / "eval_logs"),
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            deterministic=True,
        )
        callbacks.append(eval_callback)

    # 开始训练
    print(f"Starting training at {timestamp}")
    print(f"Save directory: {save_dir}")
    print(f"Total timesteps: {args.total_timesteps}")

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=CallbackList(callbacks),
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    # 保存最终模型
    final_model_path = save_dir / "final_model"
    model.save(str(final_model_path))
    print(f"Final model saved to {final_model_path}")

    env.close()
    eval_env.close()

    return model


def main():
    parser = argparse.ArgumentParser(description="Train planning-based autonomous driving with TD3")

    # 环境参数
    parser.add_argument("--env-yaml", type=str, default="env.yaml", help="Environment YAML file")
    parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel environments")

    # TD3超参数
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=256, help="Minibatch size")
    parser.add_argument("--buffer-size", type=int, default=1_000_000, help="Replay buffer size")
    parser.add_argument("--learning-starts", type=int, default=10_000, help="Random exploration steps before learning")
    parser.add_argument("--train-freq", type=int, default=1, help="Train once every N env steps")
    parser.add_argument("--gradient-steps", type=int, default=1, help="Gradient steps per update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update coefficient")
    parser.add_argument("--policy-delay", type=int, default=2, help="Delayed actor update frequency")
    parser.add_argument("--action-noise-std", type=float, default=0.1, help="Std of Gaussian exploration noise")
    parser.add_argument("--target-policy-noise", type=float, default=0.2, help="Target policy smoothing noise")
    parser.add_argument("--target-noise-clip", type=float, default=0.5, help="Target policy noise clip")

    # 训练参数
    parser.add_argument("--total-timesteps", type=int, default=1_000_000, help="Total timesteps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")

    # 保存和评估
    parser.add_argument("--save-dir", type=str, default="./logs", help="Save directory")
    parser.add_argument("--save-freq", type=int, default=10000, help="Save frequency")
    parser.add_argument("--eval-freq", type=int, default=5000, help="Evaluation frequency")
    parser.add_argument("--n-eval-episodes", type=int, default=5, help="Number of evaluation episodes")

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
