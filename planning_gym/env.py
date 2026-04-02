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
from typing import Optional, Tuple, Dict
from pathlib import Path
from scipy.optimize import minimize

try:
    import yaml
except Exception:
    yaml = None


def resolve_env_yaml_path(env_yaml: str) -> Path:
    """解析 env.yaml 路径，优先仓库根目录，其次当前工作目录。"""
    env_path = Path(env_yaml)
    if env_path.is_absolute():
        return env_path

    repo_root_path = Path(__file__).parent.parent / env_path
    if repo_root_path.exists():
        return repo_root_path

    cwd_path = Path.cwd() / env_path
    if cwd_path.exists():
        return cwd_path

    return repo_root_path


def load_env_yaml_data(env_yaml: str) -> Dict:
    """读取 env.yaml 内容。"""
    if yaml is None:
        return {}

    env_path = resolve_env_yaml_path(env_yaml)
    if not env_path.exists():
        return {}

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_global_path_config_from_yaml(env_yaml: str) -> Dict:
    data = load_env_yaml_data(env_yaml)
    gui_cfg = data.get("gui", {}) if isinstance(data, dict) else {}
    cfg = gui_cfg.get("global_path", {}) if isinstance(gui_cfg, dict) else {}
    return cfg if isinstance(cfg, dict) else {}


def load_randomization_config_from_yaml(env_yaml: str) -> Dict:
    data = load_env_yaml_data(env_yaml)
    gui_cfg = data.get("gui", {}) if isinstance(data, dict) else {}
    cfg = gui_cfg.get("randomization", {}) if isinstance(gui_cfg, dict) else {}
    return cfg if isinstance(cfg, dict) else {}


def load_obstacle_avoidance_config_from_yaml(env_yaml: str) -> Dict:
    data = load_env_yaml_data(env_yaml)
    gui_cfg = data.get("gui", {}) if isinstance(data, dict) else {}
    cfg = gui_cfg.get("obstacle_avoidance", {}) if isinstance(gui_cfg, dict) else {}
    return cfg if isinstance(cfg, dict) else {}


def load_lidar_config_from_yaml(env_yaml: str) -> Dict:
    data = load_env_yaml_data(env_yaml)
    robots = data.get("robot", []) if isinstance(data, dict) else []
    if not isinstance(robots, list) or len(robots) == 0 or not isinstance(robots[0], dict):
        return {}

    sensors = robots[0].get("sensors", [])
    if not isinstance(sensors, list):
        return {}

    for sensor in sensors:
        if isinstance(sensor, dict) and sensor.get("type") == "lidar2d":
            return sensor

    return {}


def load_robot_shape_config_from_yaml(env_yaml: str) -> Dict:
    data = load_env_yaml_data(env_yaml)
    robots = data.get("robot", []) if isinstance(data, dict) else []
    if not isinstance(robots, list) or len(robots) == 0 or not isinstance(robots[0], dict):
        return {}

    shape_cfg = robots[0].get("shape", {})
    return shape_cfg if isinstance(shape_cfg, dict) else {}


class VehicleKinematics:
    """阿克曼转向车辆运动学模型"""

    def __init__(self, wheelbase: float = 1.75, dt: float = 0.1,
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
        wheelbase: float = 1.75,
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
        horizon: int = 10,
        target_velocity: float = 3.0,
        collision_penalty: float = -100.0,
        goal_reward: float = 100.0,
        step_reward: float = 0.1,
        lidar_range_max: float = 10.0,
        lidar_points: int = 100,
        wheelbase: float = 1.75,
        forward_only: bool = True,
        v_max: float = 3.0,
        delta_max: float = 1.0,
        randomize_every_reset: Optional[bool] = None,
        min_start_goal_distance: Optional[float] = None,
        random_margin: Optional[float] = None,
        randomize_obstacles: Optional[bool] = None,
    ):
        self.env_yaml = env_yaml
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.horizon = horizon
        self.target_velocity = target_velocity
        self.collision_penalty = collision_penalty
        self.goal_reward = goal_reward
        self.step_reward = step_reward
        lidar_cfg = load_lidar_config_from_yaml(env_yaml)
        self.lidar_range_max = float(lidar_cfg.get("range_max", lidar_range_max))
        self.lidar_points = int(lidar_cfg.get("number", lidar_points))
        obstacle_cfg = load_obstacle_avoidance_config_from_yaml(env_yaml)
        self.obstacle_min_distance = float(obstacle_cfg.get("min_distance", 3.0))
        self.obstacle_danger_distance = float(obstacle_cfg.get("danger_distance", 1.0))
        shape_cfg = load_robot_shape_config_from_yaml(env_yaml)
        self.wheelbase = float(shape_cfg.get("wheelbase", wheelbase))
        self.forward_only = forward_only
        self.v_max = v_max
        self.delta_max = delta_max

        self._global_path_cfg = self._load_global_path_config()
        random_cfg = self._load_randomization_config()
        self.randomize_every_reset = (
            bool(random_cfg.get("enabled", False))
            if randomize_every_reset is None
            else bool(randomize_every_reset)
        )
        self.min_start_goal_distance = (
            float(random_cfg.get("min_start_goal_distance", 10.0))
            if min_start_goal_distance is None
            else float(min_start_goal_distance)
        )
        self.random_margin = (
            float(random_cfg.get("random_margin", 1.5))
            if random_margin is None
            else float(random_margin)
        )
        self.randomize_obstacles = (
            bool(random_cfg.get("randomize_obstacles", True))
            if randomize_obstacles is None
            else bool(randomize_obstacles)
        )
        self.max_randomization_attempts = int(random_cfg.get("max_sampling_attempts", 200))

        # 状态: [v, delta_theta, delta_y, delta, lidar...]
        self.state_dim = 4 + self.lidar_points
        # 动作: [acceleration, steering_angle]
        self.action_dim = 2

        # 根据forward_only设置速度范围
        v_min = 0.0 if forward_only else -v_max

        self.kinematics = VehicleKinematics(
            wheelbase=self.wheelbase, dt=0.1, v_min=v_min, v_max=v_max, delta_max=delta_max
        )
        self.mpc = MPCTracker(
            horizon=horizon, dt=0.1, wheelbase=self.wheelbase, v_min=v_min, v_max=v_max, delta_max=delta_max
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
        self._random_obstacle_ids = []

    def _load_randomization_config(self) -> Dict:
        """从 env.yaml 读取随机化配置"""
        return load_randomization_config_from_yaml(self.env_yaml)

    def _load_global_path_config(self) -> Dict:
        """从 env.yaml 读取全局路径配置"""
        return load_global_path_config_from_yaml(self.env_yaml)

    def _load_global_path_from_csv(self) -> Optional[np.ndarray]:
        """从CSV读取全局路径，返回(N,4): [x, y, yaw, v]"""
        cfg = self._global_path_cfg or {}
        csv_path = cfg.get("csv_path", "")
        if not csv_path:
            return None

        csv_file = Path(str(csv_path))
        if not csv_file.is_absolute():
            env_path = resolve_env_yaml_path(self.env_yaml)
            csv_file = (env_path.parent / csv_file).resolve()

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

    def _init_env(self):
        """初始化ir-sim环境"""
        if self._initialized and self.env is not None:
            return

        import irsim
        env_path = Path(__file__).parent.parent / self.env_yaml
        display = self.render_mode == "human"
        self.env = irsim.make(str(env_path), display=display, log_level="WARNING")

        # 仅随机化圆形障碍物（保留边界墙体）
        self._random_obstacle_ids = [
            obs.id for obs in self.env.obstacle_list if getattr(obs, "shape", "") == "circle"
        ]

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

    def _is_collision_free_state(self, robot, candidate_state: np.ndarray) -> bool:
        """检查候选状态是否无碰撞"""
        original_state = robot.state.copy()
        robot.set_state(candidate_state, init=False)
        collision = any(robot.check_collision(obj) for obj in self.env.obstacle_list)
        robot.set_state(original_state, init=False)
        return not collision

    def _sample_free_robot_state(self, robot, width: float, height: float) -> Optional[np.ndarray]:
        """采样无碰撞起点"""
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

    def _sample_free_goal(self, robot, width: float, height: float) -> Optional[np.ndarray]:
        """采样无碰撞终点"""
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
        """每回合随机障碍物 + 随机起终点（距离约束）"""
        robot = self.env.robot
        width = float(getattr(self.env.world_param, "width", 37.0))
        height = float(getattr(self.env.world_param, "height", 20.0))

        # 1) 随机障碍物（仅圆障碍）
        if self.randomize_obstacles and len(self._random_obstacle_ids) > 0:
            low = [self.random_margin, self.random_margin, -np.pi]
            high = [max(self.random_margin + 0.1, width - self.random_margin),
                    max(self.random_margin + 0.1, height - self.random_margin),
                    np.pi]
            self.env.random_obstacle_position(low, high, ids=self._random_obstacle_ids, non_overlapping=True)

        # 2) 随机起终点并满足距离约束
        selected_start = None
        selected_goal = None
        for _ in range(self.max_randomization_attempts):
            start_state = self._sample_free_robot_state(robot, width, height)
            goal = self._sample_free_goal(robot, width, height)
            if start_state is None or goal is None:
                continue

            dist = np.linalg.norm(start_state[:2, 0] - goal[:2])
            if dist >= self.min_start_goal_distance:
                selected_start = start_state
                selected_goal = goal
                break

        # 回退：保留当前布局
        if selected_start is None or selected_goal is None:
            return

        robot.set_state(selected_start, init=True)
        if robot.velocity is not None:
            robot.set_velocity(np.zeros_like(robot.velocity), init=True)
        robot.set_goal(selected_goal.tolist(), init=True)
        self.env.build_tree()

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
        """将车辆初始状态和目标点对齐到CSV轨迹首尾。"""
        if self.global_path is None or len(self.global_path) < 2:
            return

        robot = self.env.robot

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
        self.env.build_tree()

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
            if min_distance < self.obstacle_min_distance:
                obstacle_penalty = (self.obstacle_min_distance - min_distance) * 1.0
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

        if self.randomize_every_reset:
            self._randomize_episode_layout()

        self.env.reset()

        # 每回合重建起终点和全局路径
        robot = self.env.robot
        self.start_position = robot.state[:2].flatten().copy()
        if robot.goal is not None:
            self.goal_position = robot.goal[:2].flatten().copy()
        self._generate_global_path()
        self._apply_csv_start_goal_if_needed()

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
        a_max = 3.0  # m/s^2
        acceleration = action[0] * a_max  # [-3, 3] m/s^2
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
