"""
规划型自动驾驶PPO训练脚本
"""

import os
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from gymnasium import Env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

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
    save_dir = Path(args.save_dir) / f"planning_ppo_{timestamp}"
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

    # 创建PPO模型
    model = PPO(
        "MlpPolicy",
        env,
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

    # 创建回调
    callbacks = []

    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(save_dir / "checkpoints"),
        name_prefix="planning_ppo",
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
    parser = argparse.ArgumentParser(description="Train planning-based autonomous driving with PPO")

    # 环境参数
    parser.add_argument("--env-yaml", type=str, default="env.yaml", help="Environment YAML file")
    parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel environments")

    # PPO超参数
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=2048, help="Number of steps per update")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--n-epochs", type=int, default=10, help="Number of epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Max gradient norm")

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