"""
训练回调模块

提供成功率等自定义指标的跟踪功能
"""

import numpy as np
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback


class SuccessRateCallback(BaseCallback):
    """
    成功率跟踪回调

    跟踪智能体到达终点的成功率（滚动窗口）
    成功定义: info["arrive"] == True (到达终点)
    """

    def __init__(self, verbose: int = 0, window_size: int = 100):
        super().__init__(verbose)
        self.window_size = window_size
        self.episode_results = deque(maxlen=window_size)  # 1=成功, 0=失败
        self.total_episodes = 0
        self.total_success = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for info, done in zip(infos, dones):
            if done:
                self.total_episodes += 1
                if info.get("arrive", False):
                    self.episode_results.append(1)
                    self.total_success += 1
                else:
                    self.episode_results.append(0)

        return True

    def _on_rollout_end(self) -> None:
        # 滚动窗口成功率
        if len(self.episode_results) > 0:
            window_success_rate = sum(self.episode_results) / len(self.episode_results)
            self.logger.record("rollout/success_rate", window_success_rate)

        # 累计成功率
        if self.total_episodes > 0:
            total_success_rate = self.total_success / self.total_episodes
            self.logger.record("rollout/total_success_rate", total_success_rate)
            self.logger.record("rollout/total_episodes", self.total_episodes)


class CollisionRateCallback(BaseCallback):
    """
    碰撞率跟踪回调（滚动窗口）
    """

    def __init__(self, verbose: int = 0, window_size: int = 100):
        super().__init__(verbose)
        self.window_size = window_size
        self.episode_results = deque(maxlen=window_size)
        self.total_episodes = 0
        self.total_collisions = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for info, done in zip(infos, dones):
            if done:
                self.total_episodes += 1
                if info.get("collision", False):
                    self.episode_results.append(1)
                    self.total_collisions += 1
                else:
                    self.episode_results.append(0)

        return True

    def _on_rollout_end(self) -> None:
        if len(self.episode_results) > 0:
            window_collision_rate = sum(self.episode_results) / len(self.episode_results)
            self.logger.record("rollout/collision_rate", window_collision_rate)


class DetailedMetricsCallback(BaseCallback):
    """
    详细指标跟踪回调（滚动窗口）

    同时跟踪成功率、碰撞率、超时率
    默认使用100个episode的滚动窗口
    """

    def __init__(self, verbose: int = 0, window_size: int = 100):
        super().__init__(verbose)
        self.window_size = window_size
        # 滚动窗口: 'success', 'collision', 'timeout'
        self.episode_results = deque(maxlen=window_size)
        # 累计统计
        self.total_episodes = 0
        self.total_success = 0
        self.total_collision = 0
        self.total_timeout = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for info, done in zip(infos, dones):
            if done:
                self.total_episodes += 1

                if info.get("arrive", False):
                    self.episode_results.append("success")
                    self.total_success += 1
                elif info.get("collision", False):
                    self.episode_results.append("collision")
                    self.total_collision += 1
                else:
                    self.episode_results.append("timeout")
                    self.total_timeout += 1

        return True

    def _on_rollout_end(self) -> None:
        if len(self.episode_results) > 0:
            # 滚动窗口指标
            window_success = sum(1 for r in self.episode_results if r == "success")
            window_collision = sum(1 for r in self.episode_results if r == "collision")
            window_timeout = sum(1 for r in self.episode_results if r == "timeout")

            n = len(self.episode_results)
            self.logger.record("rollout/success_rate", window_success / n)
            self.logger.record("rollout/collision_rate", window_collision / n)
            self.logger.record("rollout/timeout_rate", window_timeout / n)
            self.logger.record("rollout/total_episodes", self.total_episodes)

    def get_summary(self) -> dict:
        """获取统计摘要"""
        return {
            "total_episodes": self.total_episodes,
            "success_count": self.total_success,
            "collision_count": self.total_collision,
            "timeout_count": self.total_timeout,
            "success_rate": self.total_success / max(self.total_episodes, 1),
            "collision_rate": self.total_collision / max(self.total_episodes, 1),
            "timeout_rate": self.total_timeout / max(self.total_episodes, 1),
        }