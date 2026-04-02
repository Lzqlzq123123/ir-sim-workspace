"""
规划型自动驾驶训练配置
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PlanningConfig:
    """规划型训练配置"""

    # 环境参数
    env_yaml: str = "env.yaml"
    max_steps: int = 1000
    target_velocity: float = 5.0
    lidar_points: int = 100

    # MPC参数
    horizon: int = 10
    trajectory_dim: int = 5
    wheelbase: float = 3.0

    # 奖励参数
    collision_penalty: float = -100.0
    goal_reward: float = 100.0
    step_reward: float = 0.1

    # PPO参数
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # 训练参数
    total_timesteps: int = 1_000_000
    n_envs: int = 4
    seed: Optional[int] = None
    device: str = "auto"

    # 保存参数
    save_dir: str = "./logs"
    save_freq: int = 10000
    eval_freq: int = 5000
    n_eval_episodes: int = 5


# 默认配置
DEFAULT_CONFIG = PlanningConfig()


# 高性能配置
HIGH_PERFORMANCE_CONFIG = PlanningConfig(
    n_envs=8,
    batch_size=256,
    n_steps=4096,
    total_timesteps=10_000_000,
)


# 快速测试配置
QUICK_TEST_CONFIG = PlanningConfig(
    n_envs=1,
    total_timesteps=10000,
    n_steps=256,
    batch_size=32,
)