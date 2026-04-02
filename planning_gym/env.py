"""
规划型自动驾驶RL环境

模型输入:
- 自车当前状态: 速度v, 航向角偏差Δθ, 横向偏差Δy, 前轮转角δ
- 障碍物状态: 自车周围激光点云分布

模型输出:
- 目标加速度 a 和目标前轮转角 δ
- 通过车辆运动学模型预测轨迹
- MPC轨迹跟踪器跟踪预测轨迹
"""

import numpy as np
import yaml
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
from scipy.optimize import minimize


def resolve_env_yaml_path(env_yaml: str) -> Path:
    """解析环境 YAML 的实际路径。"""
    env_path = Path(env_yaml)
    if not env_path.is_absolute():
        env_path = Path(__file__).parent.parent / env_path
    return env_path.resolve()


def load_env_config(env_yaml: str) -> Dict[str, Any]:
    """从环境 YAML 中提取规划相关配置。"""
    env_path = resolve_env_yaml_path(env_yaml)
    with env_path.open("r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f) or {}

    robot_cfg = (raw_config.get("robot") or [{}])[0]
    shape_cfg = robot_cfg.get("shape") or {}
    vel_min_cfg = robot_cfg.get("vel_min") or [0.0, -1.0]
    vel_max_cfg = robot_cfg.get("vel_max") or [3.0, 1.0]
    sensors_cfg = robot_cfg.get("sensors") or []
    lidar_cfg = next((sensor for sensor in sensors_cfg if sensor.get("type") == "lidar2d"), {})
    world_cfg = raw_config.get("world") or {}
    plot_cfg = world_cfg.get("plot") or {}
    planning_cfg = plot_cfg.get("planning") or {}
    reward_cfg = (
        raw_config.get("reward")
        or plot_cfg.get("reward")
        or {}
    )

    def _float_pair(values, index, default):
        try:
            return float(values[index])
        except (IndexError, TypeError, ValueError):
            return default

    return {
        "path": env_path,
        "step_time": float(world_cfg.get("step_time", 0.1)),
        "sample_time": float(world_cfg.get("sample_time", world_cfg.get("step_time", 0.1))),
        "vehicle_length": float(shape_cfg.get("length", 4.6)),
        "vehicle_width": float(shape_cfg.get("width", 1.6)),
        "wheelbase": float(shape_cfg.get("wheelbase", 3.0)),
        "v_min": _float_pair(vel_min_cfg, 0, 0.0),
        "delta_min": _float_pair(vel_min_cfg, 1, -1.0),
        "v_max": _float_pair(vel_max_cfg, 0, 3.0),
        "delta_max": abs(_float_pair(vel_max_cfg, 1, 1.0)),
        "target_velocity": _float_pair(vel_max_cfg, 0, 3.0),
        "lidar_range_max": float(lidar_cfg.get("range_max", 10.0)),
        "lidar_points": int(lidar_cfg.get("number", 100)),
        "planning_horizon": int(planning_cfg.get("horizon", 10)),
        "obstacle_penalty_distance": float(reward_cfg.get("obstacle_penalty_distance", 3.0)),
        "obstacle_danger_distance": float(reward_cfg.get("obstacle_danger_distance", 1.0)),
    }


class VehicleKinematics:
    """阿克曼转向车辆运动学模型"""

    def __init__(self, wheelbase: float = 3.0, dt: float = 0.1,
                 v_max: float = 8.0, v_min: float = 0.0,
                 delta_max: float = 1.0, a_max: float = 3.0):
        self.wheelbase = wheelbase
        self.dt = dt
        self.v_max = v_max
        self.v_min = v_min  # 默认为0，只允许前进
        self.delta_max = delta_max
        self.a_max = a_max

    def predict_trajectory(self, state: np.ndarray, acceleration: float,
                           steering: float, horizon: int) -> np.ndarray:
        """
        预测未来轨迹

        Args:
            state: [x, y, theta, v] 当前状态
            acceleration: 目标加速度 (m/s^2)
            steering: 目标前轮转角 (rad)
            horizon: 预测步数

        Returns:
            trajectory: (horizon+1, 4) 预测轨迹 [x, y, theta, v]
        """
        trajectory = np.zeros((horizon + 1, 4))
        trajectory[0] = state.copy()

        x, y, theta, v = state

        for i in range(horizon):
            # 更新速度（限制最小速度为v_min，防止倒车）
            v = v + acceleration * self.dt
            v = np.clip(v, self.v_min, self.v_max)

            # 限制转角
            delta = np.clip(steering, -self.delta_max, self.delta_max)

            # 阿克曼运动学
            x = x + v * np.cos(theta) * self.dt
            y = y + v * np.sin(theta) * self.dt
            theta = theta + v * np.tan(delta) / self.wheelbase * self.dt
            theta = self._normalize_angle(theta)

            trajectory[i + 1] = [x, y, theta, v]

        return trajectory

    def _normalize_angle(self, angle: float) -> float:
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


class MPCTracker:
    """MPC轨迹跟踪控制器

    使用模型预测控制跟踪参考轨迹：
    预测多步状态，优化控制量使预测状态接近参考轨迹
    """

    def __init__(
        self,
        horizon: int = 10,
        dt: float = 0.1,
        wheelbase: float = 3.0,
        v_max: float = 8.0,
        v_min: float = 0.0,
        delta_max: float = 1.0,
        Q: np.ndarray = None,
        R: np.ndarray = None,
    ):
        self.horizon = horizon
        self.dt = dt
        self.wheelbase = wheelbase
        self.v_max = v_max
        self.v_min = v_min
        self.delta_max = delta_max

        # 状态权重 [x, y, theta, v]
        self.Q = Q if Q is not None else np.array([1.0, 1.0, 0.5, 0.1])
        # 控制权重 [v, delta]
        self.R = R if R is not None else np.array([0.01, 0.01])

    def compute_control(
        self,
        current_state: np.ndarray,
        reference_trajectory: np.ndarray,
        trajectory_index: int = 0,
    ) -> np.ndarray:
        """计算跟踪轨迹的控制量

        Args:
            current_state: 当前状态 [x, y, theta, v]
            reference_trajectory: 参考轨迹 (N, 4) [x, y, theta, v]

        Returns:
            control: [velocity, steering_angle]
        """
        if reference_trajectory is None or len(reference_trajectory) < 2:
            return np.array([self.v_max / 2, 0.0])

        def cost_function(u):
            """MPC代价函数：预测状态与参考轨迹的误差"""
            v_cmd, delta_cmd = u

            # 预测轨迹
            pred_traj = self._predict(current_state, v_cmd, delta_cmd, self.horizon)

            # 计算与参考轨迹的误差
            cost = 0.0
            for i in range(min(self.horizon, len(reference_trajectory))):
                ref = reference_trajectory[i]
                pred = pred_traj[i]

                # 状态误差
                pos_error = (pred[0] - ref[0])**2 + (pred[1] - ref[1])**2
                theta_error = self._normalize_angle(pred[2] - ref[2])**2
                v_error = (pred[3] - ref[3])**2

                cost += self.Q[0] * pos_error + self.Q[2] * theta_error + self.Q[3] * v_error

            # 控制量惩罚
            cost += self.R[0] * v_cmd**2 + self.R[1] * delta_cmd**2

            return cost

        # 初始猜测
        ref_v = reference_trajectory[1][3] if len(reference_trajectory) > 1 else self.v_max / 2
        ref_theta0 = reference_trajectory[0][2]
        ref_theta1 = reference_trajectory[1][2] if len(reference_trajectory) > 1 else ref_theta0
        init_delta = self._normalize_angle(ref_theta1 - ref_theta0)

        u0 = np.array([np.clip(ref_v, self.v_min, self.v_max), init_delta])

        # 优化
        bounds = [(self.v_min, self.v_max), (-self.delta_max, self.delta_max)]
        result = minimize(
            cost_function, u0, method='SLSQP', bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-4}
        )

        return result.x

    def _predict(self, state, v_cmd, delta_cmd, steps):
        """预测未来状态"""
        traj = np.zeros((steps + 1, 4))
        traj[0] = state.copy()

        x, y, theta, v = state

        for i in range(steps):
            v = v_cmd
            delta = np.clip(delta_cmd, -self.delta_max, self.delta_max)

            x = x + v * np.cos(theta) * self.dt
            y = y + v * np.sin(theta) * self.dt
            theta = theta + v * np.tan(delta) / self.wheelbase * self.dt
            theta = self._normalize_angle(theta)

            traj[i + 1] = [x, y, theta, v]

        return traj

    def _normalize_angle(self, angle: float) -> float:
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


class PlanningEnv:
    """规划型自动驾驶RL环境"""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        env_yaml: str = "env.yaml",
        render_mode: Optional[str] = None,
        max_steps: int = 1000,
        horizon: Optional[int] = None,
        target_velocity: Optional[float] = None,
        collision_penalty: float = -100.0,
        goal_reward: float = 100.0,
        step_reward: float = 0.1,
        lidar_range_max: Optional[float] = None,
        lidar_points: Optional[int] = None,
        wheelbase: Optional[float] = None,
        forward_only: bool = True,
        v_max: Optional[float] = None,
        delta_max: Optional[float] = None,
        dt: Optional[float] = None,
        a_max: float = 3.0,
    ):
        self.env_yaml = env_yaml
        self.env_config = load_env_config(env_yaml)
        self.env_yaml_path = self.env_config["path"]
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.horizon = int(horizon if horizon is not None else self.env_config["planning_horizon"])
        self.target_velocity = float(
            target_velocity if target_velocity is not None else self.env_config["target_velocity"]
        )
        self.collision_penalty = collision_penalty
        self.goal_reward = goal_reward
        self.step_reward = step_reward
        self.lidar_range_max = float(
            lidar_range_max if lidar_range_max is not None else self.env_config["lidar_range_max"]
        )
        self.lidar_points = int(
            lidar_points if lidar_points is not None else self.env_config["lidar_points"]
        )
        self.vehicle_length = float(self.env_config["vehicle_length"])
        self.vehicle_width = float(self.env_config["vehicle_width"])
        self.wheelbase = float(wheelbase if wheelbase is not None else self.env_config["wheelbase"])
        self.forward_only = forward_only
        self.v_max = float(v_max if v_max is not None else self.env_config["v_max"])
        self.delta_max = float(delta_max if delta_max is not None else self.env_config["delta_max"])
        self.obstacle_penalty_distance = float(self.env_config["obstacle_penalty_distance"])
        self.obstacle_danger_distance = float(self.env_config["obstacle_danger_distance"])
        self.dt = float(dt if dt is not None else self.env_config["step_time"])
        self.a_max = a_max

        # 状态: [v, delta_theta, delta_y, delta, lidar...]
        self.state_dim = 4 + self.lidar_points
        # 动作: [acceleration, steering_angle]
        self.action_dim = 2

        # 根据forward_only设置速度范围
        v_min = 0.0 if forward_only else -self.v_max

        self.kinematics = VehicleKinematics(
            wheelbase=self.wheelbase, dt=self.dt, v_min=v_min, v_max=self.v_max, delta_max=self.delta_max, a_max=a_max
        )
        self.mpc = MPCTracker(
            horizon=self.horizon, dt=self.dt, wheelbase=self.wheelbase, v_min=v_min, v_max=self.v_max, delta_max=self.delta_max
        )

        self.env = None
        self.current_step = 0
        self.prev_distance = None
        self.prev_control = None

        # 全局路径和局部路径
        self.goal_position = None
        self.start_position = None
        self.global_path = None
        self.local_path = None  # 局部路径（当前最近点向前的轨迹段）
        self.local_path_length = 20  # 局部轨迹点数
        self._closest_idx = 0  # 当前最近点索引
        self._initialized = False

    def _init_env(self):
        """初始化ir-sim环境"""
        if self._initialized and self.env is not None:
            return

        import irsim
        display = self.render_mode == "human"
        self.env = irsim.make(str(self.env_yaml_path), display=display, log_level="WARNING")

        robot = self.env.robot

        # 获取起点
        self.start_position = robot.state[:2].flatten().copy()

        # 获取目标点
        if robot.goal is not None:
            self.goal_position = robot.goal[:2].flatten()
        else:
            self.goal_position = np.array([50.0, 25.0])

        # 生成全局路径（从起点到终点的直线）
        self._generate_global_path()

        self.prev_control = np.array([self.v_max / 2, 0.0])
        self._initialized = True

    def _generate_global_path(self):
        """生成全局参考路径（起点到终点的直线）"""
        num_points = 100
        t = np.linspace(0, 1, num_points)

        self.global_path = np.zeros((num_points, 4))
        self.global_path[:, 0] = self.start_position[0] + t * (self.goal_position[0] - self.start_position[0])
        self.global_path[:, 1] = self.start_position[1] + t * (self.goal_position[1] - self.start_position[1])

        dx = self.goal_position[0] - self.start_position[0]
        dy = self.goal_position[1] - self.start_position[1]
        self.global_path[:, 2] = np.arctan2(dy, dx)
        self.global_path[:, 3] = self.target_velocity

    def _get_observation(self) -> np.ndarray:
        """获取当前观测"""
        robot = self.env.robot
        robot_state = robot.state.flatten()

        # 速度
        v = 0.0
        if robot.velocity is not None and len(robot.velocity) > 0:
            v = float(robot.velocity.flatten()[0])

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

        # 航向角偏差：自车航向角与局部轨迹切线方向的差值
        # 计算方式: delta_theta = theta_vehicle - theta_ref
        # theta_ref 是局部轨迹在最近点处的切线方向（航向角）
        delta_theta = self._normalize_angle(robot_state[2] - ref_point[2])

        # 横向偏差：自车位置到局部轨迹的垂直距离
        # 计算方式：将自车位置转换到参考点坐标系，取y方向偏差
        # 正值表示在轨迹左侧，负值表示在轨迹右侧
        dx = robot_state[0] - ref_point[0]
        dy = robot_state[1] - ref_point[1]
        ref_heading = ref_point[2]
        delta_y = -dx * np.sin(ref_heading) + dy * np.cos(ref_heading)

        # 前轮转角
        delta = robot_state[3] if len(robot_state) > 3 else 0.0

        # 激光雷达数据
        lidar_scan = self.env.get_lidar_scan()
        if lidar_scan is not None:
            ranges = lidar_scan.get("ranges", np.ones(self.lidar_points) * self.lidar_range_max)
            lidar_normalized = np.clip(ranges / self.lidar_range_max, 0, 1)
        else:
            lidar_normalized = np.ones(self.lidar_points)

        observation = np.concatenate([
            [v / max(self.target_velocity, 0.1)],
            [delta_theta / np.pi],
            [np.clip(delta_y / 5.0, -1, 1)],
            [np.clip(delta / 1.0, -1, 1)],
            lidar_normalized
        ])

        return observation.astype(np.float32)

    def _normalize_angle(self, angle: float) -> float:
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def _compute_reward(self, info: Dict) -> float:
        """计算奖励

        奖励设计原则：
        1. 进度奖励：向目标移动（核心）
        2. 速度奖励：保持移动
        3. 障碍物避障：远离障碍物
        4. 不强制跟踪全局路径，允许自由绕行
        """
        robot = self.env.robot
        robot_state = robot.state.flatten()

        distance_to_goal = np.sqrt(
            (robot_state[0] - self.goal_position[0])**2 +
            (robot_state[1] - self.goal_position[1])**2
        )

        reward = 0.0

        # === 1. 进度奖励（核心）：向目标移动 ===
        if self.prev_distance is not None:
            progress = self.prev_distance - distance_to_goal
            reward += progress * 5.0

        self.prev_distance = distance_to_goal

        # === 2. 速度奖励：保持移动 ===
        v = 0.0
        if robot.velocity is not None and len(robot.velocity) > 0:
            v = float(robot.velocity.flatten()[0])

        if v > 1.0:
            reward += 0.5
        elif v > 0.5:
            reward += 0.2
        elif v < 0.1:
            reward -= 1.0  # 停止惩罚

        # === 3. 障碍物避障（关键！）===
        lidar_scan = self.env.get_lidar_scan()
        if lidar_scan is not None:
            ranges = lidar_scan.get("ranges", np.ones(self.lidar_points) * self.lidar_range_max)
            min_distance = np.min(ranges)

            # 障碍物越近，惩罚越大
            if min_distance < self.obstacle_penalty_distance:
                obstacle_penalty = (self.obstacle_penalty_distance - min_distance) * 1.0
                reward -= obstacle_penalty
            if min_distance < self.obstacle_danger_distance:
                reward -= 2.0  # 危险距离

        # === 4. 全局路径参考（可选，不强制）===
        # 只在远离目标时轻微引导
        if distance_to_goal > 5.0:
            # 轻微引导朝向目标方向
            dx = self.goal_position[0] - robot_state[0]
            dy = self.goal_position[1] - robot_state[1]
            goal_heading = np.arctan2(dy, dx)
            heading_to_goal = abs(self._normalize_angle(goal_heading - robot_state[2]))
            # 轻微惩罚：偏离目标方向太多时
            if heading_to_goal > np.pi / 2:  # 偏离超过90度
                reward -= 0.1

        # === 5. 碰撞/到达 ===
        if robot.collision_flag:
            reward += self.collision_penalty  # -100
        if robot.arrive_flag:
            reward += self.goal_reward  # 100

        return reward

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)

        if self.env is None:
            self._init_env()
        else:
            self.env.reset()

        self.current_step = 0
        self.prev_distance = None
        self.prev_control = np.array([self.v_max / 2, 0.0])

        observation = self._get_observation()
        return observation, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一步动作

        Args:
            action: [acceleration, steering_angle] 归一化到 [-1, 1]
                   acceleration: -1 表示最大减速, 1 表示最大加速
                   steering: -1 表示最大左转, 1 表示最大右转

        工作流程：
            1. 模型输出 [加速度, 转角]
            2. 运动学模型预测轨迹
            3. MPC跟踪轨迹，输出控制量 [速度, 转角]
        """
        # 将归一化动作转换为实际值
        acceleration = action[0] * self.a_max
        steering = action[1] * self.delta_max

        # 获取当前状态
        robot = self.env.robot
        v = 0.0
        if robot.velocity is not None and len(robot.velocity) > 0:
            v = float(robot.velocity.flatten()[0])

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

        # 执行控制
        self.env.step(control)

        self.current_step += 1

        # 获取观测
        observation = self._get_observation()

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

        # 计算奖励
        reward = self._compute_reward(info)

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human" and self.env is not None:
            self.env.render()

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
            self._initialized = False

    @property
    def observation_space(self):
        from gymnasium import spaces
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )

    @property
    def action_space(self):
        from gymnasium import spaces
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32
        )
