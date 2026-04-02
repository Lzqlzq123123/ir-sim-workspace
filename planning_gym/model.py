"""
规划型自动驾驶神经网络模型

模型输出轨迹相关参数，然后由MPC跟踪器跟踪轨迹
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
from typing import Tuple, Dict, Optional


class PlanningEncoder(nn.Module):
    """
    规划型编码器

    编码自车状态和激光点云数据
    """

    def __init__(
        self,
        vehicle_state_dim: int = 4,
        lidar_points: int = 100,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # 自车状态编码
        self.vehicle_encoder = nn.Sequential(
            nn.Linear(vehicle_state_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4),
            nn.ReLU(),
        )

        # 激光雷达编码
        self.lidar_encoder = nn.Sequential(
            nn.Linear(lidar_points, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim // 4 + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, vehicle_state: torch.Tensor, lidar_data: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            vehicle_state: 自车状态 [batch, 4]
            lidar_data: 激光数据 [batch, lidar_points]

        Returns:
            fused_features: 融合特征 [batch, hidden_dim]
        """
        vehicle_feat = self.vehicle_encoder(vehicle_state)
        lidar_feat = self.lidar_encoder(lidar_data)
        fused = torch.cat([vehicle_feat, lidar_feat], dim=-1)
        return self.fusion(fused)


class TrajectoryDecoder(nn.Module):
    """
    轨迹解码器

    从特征解码轨迹参数
    """

    def __init__(
        self,
        feature_dim: int,
        trajectory_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, trajectory_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        解码轨迹参数

        Args:
            features: 输入特征 [batch, feature_dim]

        Returns:
            trajectory_params: 轨迹参数 [batch, trajectory_dim]
        """
        return self.decoder(features)


class PlanningActorCritic(nn.Module):
    """
    规划型Actor-Critic网络

    输出轨迹参数，用于生成参考轨迹供MPC跟踪
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lidar_points: int = 100,
        hidden_dim: int = 256,
        use_attention: bool = False,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lidar_points = lidar_points
        self.use_attention = use_attention

        # 编码器
        self.encoder = PlanningEncoder(
            vehicle_state_dim=4,
            lidar_points=lidar_points,
            hidden_dim=hidden_dim,
        )

        # 注意力机制 (可选)
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                batch_first=True,
            )

        # Actor头 (轨迹参数)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh(),  # 输出范围 [-1, 1]
        )

        # 动作标准差
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic头
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            obs: 观测 [batch, state_dim]

        Returns:
            action_mean: 动作均值
            action_std: 动作标准差
            value: 状态价值
        """
        # 分离自车状态和激光数据
        vehicle_state = obs[..., :4]
        lidar_data = obs[..., 4:]

        # 编码
        features = self.encoder(vehicle_state, lidar_data)

        # 注意力 (可选)
        if self.use_attention:
            features = features.unsqueeze(1)  # [batch, 1, hidden]
            features, _ = self.attention(features, features, features)
            features = features.squeeze(1)

        # Actor输出
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_log_std.expand_as(action_mean))

        # Critic输出
        value = self.critic(features)

        return action_mean, action_std, value

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取动作

        Args:
            obs: 观测
            deterministic: 是否确定性策略

        Returns:
            action: 动作
            log_prob: 对数概率
            value: 价值
        """
        action_mean, action_std, value = self.forward(obs)

        if deterministic:
            action = action_mean
            log_prob = torch.zeros(obs.shape[0], device=obs.device)
        else:
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob, value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估动作

        Args:
            obs: 观测
            actions: 动作

        Returns:
            log_prob: 对数概率
            entropy: 熵
            value: 价值
        """
        action_mean, action_std, value = self.forward(obs)

        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy, value


class PlanningPolicyWithCost(nn.Module):
    """
    带代价感知的规划策略网络

    除了轨迹参数外，还输出预测的代价/风险
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lidar_points: int = 100,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.actor_critic = PlanningActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            lidar_points=lidar_points,
            hidden_dim=hidden_dim,
        )

        # 代价预测头
        self.cost_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),  # 输出范围 [0, 1]
        )

    def forward(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            obs: 观测

        Returns:
            action_mean: 动作均值
            action_std: 动作标准差
            value: 状态价值
            predicted_cost: 预测代价
        """
        # 获取中间特征
        vehicle_state = obs[..., :4]
        lidar_data = obs[..., 4:]
        features = self.actor_critic.encoder(vehicle_state, lidar_data)

        # Actor-Critic
        action_mean = self.actor_critic.actor_mean(features)
        action_std = torch.exp(self.actor_critic.actor_log_std.expand_as(action_mean))
        value = self.actor_critic.critic(features)

        # 代价预测
        predicted_cost = self.cost_predictor(features)

        return action_mean, action_std, value, predicted_cost


if __name__ == "__main__":
    # 测试网络
    batch_size = 4
    lidar_points = 100
    state_dim = 4 + lidar_points
    action_dim = 5  # 轨迹参数维度

    model = PlanningActorCritic(state_dim, action_dim, lidar_points)
    obs = torch.randn(batch_size, state_dim)

    action, log_prob, value = model.get_action(obs)
    print(f"Action shape: {action.shape}")
    print(f"Log prob shape: {log_prob.shape}")
    print(f"Value shape: {value.shape}")

    # 测试带代价的网络
    model_with_cost = PlanningPolicyWithCost(state_dim, action_dim, lidar_points)
    action_mean, action_std, value, cost = model_with_cost(obs)
    print(f"Predicted cost shape: {cost.shape}")

    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")