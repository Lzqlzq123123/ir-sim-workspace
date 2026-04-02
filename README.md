# NeuPAN 自动驾驶训练环境

本工作空间使用 Stable Baselines3 的 TD3 算法进行规划型自动驾驶训练。

---

## 安装

### 依赖

- Python >= 3.10
- [uv](https://github.com/astral-sh/uv)（包管理工具）

### 安装步骤

**1. 安装 uv（如果尚未安装）**

```bash
pip install uv
```

**2. 在 workspace 目录下创建虚拟环境**

```bash
cd /path/to/model_train/workspace
uv venv .venv --python 3.11
```

**3. 激活虚拟环境**

```bash
source .venv/bin/activate
```

**4. 安装依赖**

```bash
uv pip install numpy torch gymnasium stable-baselines3 tensorboard tqdm rich scipy matplotlib
```

**5. 安装仿真环境 ir-sim（本地可编辑安装）**

```bash
uv pip install -e ../ir-sim
```

**6. 验证安装**

```bash
python -c "import irsim; import stable_baselines3; import torch; print('OK')"
```

> **注意**：如果可视化报 TkAgg 错误，说明 uv 管理的 Python 不含 Tk，执行以下命令修复：
> ```bash
> pip install --upgrade uv
> uv python upgrade --reinstall
> # 然后重新执行步骤 2~5
> ```

---

## 目录结构

```
workspace/
├── env.yaml                  # 环境配置文件（37m x 20m 矩形区域）
├── README.md                 # 使用说明
├── train_control.py          # 控制型训练脚本（支持实时可视化）
├── train_planning.py         # 规划型训练脚本（支持实时可视化）
├── eval_control.py           # 控制型评估脚本
├── eval_planning.py          # 规划型评估脚本
├── control_gym/              # 控制型模块
│   ├── env.py                # Gym环境封装
│   ├── model.py              # 神经网络模型
│   └── config.py             # 配置
└── planning_gym/             # 规划型模块
    ├── env.py                # Gym环境 + 运动学模型 + MPC跟踪器
    ├── model.py              # 神经网络模型
    └── config.py             # 配置
```

---

## 环境说明

### 环境配置 (env.yaml)

| 参数 | 值 | 说明 |
|------|-----|------|
| 世界范围 | 37m × 20m | 矩形区域 |
| 起点 | (2, 10) | 左侧中间 |
| 终点 | (35, 10) | 右侧中间 |
| 目标速度 | 3 m/s | 最大速度 |
| 障碍物 | 4个圆形 | 半径1.2m |
| 边界 | 四面墙壁 | 防止车辆跑出 |

### 障碍物位置

| 编号 | 位置 (x, y) |
|------|-------------|
| 1 | (10, 7) |
| 2 | (15, 10) |
| 3 | (24, 7) |
| 4 | (20, 15) |

### 状态空间 (104维)

| 维度 | 含义 | 范围 | 说明 |
|------|------|------|------|
| 0 | 速度 | [0, 1] | v / target_velocity |
| 1 | 航向角偏差 | [-1, 1] | Δθ / π |
| 2 | 横向偏差 | [-1, 1] | Δy / 5.0 |
| 3 | 前轮转角 | [-1, 1] | δ |
| 4-103 | 激光雷达 | [0, 1] | 100个点的归一化距离 |

---

## 奖励函数设计

### 奖励目标

**核心目标**: 让智能体学会从起点安全、高效地到达终点。

具体目标：
1. **到达终点**: 鼓励智能体最终到达目标位置
2. **避免碰撞**: 防止智能体与障碍物碰撞
3. **高效前进**: 鼓励智能体快速向目标移动
4. **保持航向**: 鼓励智能体朝向目标方向行驶

### 控制型奖励函数

```python
reward = step_reward                                    # 步数奖励: +0.1

# 1. 进度奖励（主要驱动力）
reward += (prev_distance - distance_to_goal) * 2.0      # 靠近目标: 正奖励

# 2. 速度奖励（保持目标速度）
reward -= 1.0 * |v - target_velocity| / target_velocity # 速度偏差惩罚

# 3. 航向奖励（朝向目标）
reward -= 0.5 * heading_error / π                       # 航向偏差惩罚

# 4. 横向偏差惩罚（沿路径行驶）
reward -= 0.3 * lateral_error / 5.0                     # 偏离路径惩罚

# 5. 终止奖励
reward += -100.0    # 碰撞
reward += +100.0    # 到达终点
```

### 规划型奖励函数

```python
reward = step_reward                                    # 步数奖励: +0.1

# 1. 进度奖励（主要驱动力）
reward += (prev_distance - distance_to_goal) * 2.0      # 靠近目标: 正奖励

# 2. 航向奖励（朝向目标）
reward -= 0.3 * heading_error / π                       # 航向偏差惩罚

# 3. 终止奖励
reward += -100.0    # 碰撞
reward += +100.0    # 到达终点
```

### 奖励对比

| 奖励项 | 控制型 | 规划型 | 说明 |
|--------|--------|--------|------|
| 步数奖励 | +0.1 | +0.1 | 鼓励存活 |
| 进度奖励 | progress × 2.0 | progress × 2.0 | 主要驱动力 |
| 速度奖励 | ✓ | ✗ | 控制型要求保持速度 |
| 航向奖励 | -0.5 × error/π | -0.3 × error/π | 朝向目标 |
| 横向偏差 | -0.3 × error/5.0 | ✗ | 控制型要求沿路径 |
| 碰撞惩罚 | -100 | -100 | 避免碰撞 |
| 到达奖励 | +100 | +100 | 达成目标 |

**差异说明**:
- 控制型多了速度奖励和横向偏差惩罚，要求更精确的控制
- 规划型奖励更简洁，主要关注进度和航向

---

## 环境要求

```bash
# 使用conda环境
conda activate py310

# 安装依赖
pip install stable-baselines3 gymnasium numpy scipy shapely pyyaml imageio loguru tqdm rich

# 可视化需要交互式后端
pip install tk  # 或 PyQt5
```

---

## 快速开始

```bash
# 激活虚拟环境
source .venv/bin/activate

# 控制型训练
MPLBACKEND=Agg python train_control.py --total-timesteps 100000

# 规划型训练
MPLBACKEND=Agg python train_planning.py --total-timesteps 100000

# 带实时可视化训练
python train_control.py --total-timesteps 100000 --visualize
python train_planning.py --total-timesteps 100000 --visualize
```

---

## 方法一：控制型 (control_gym)

模型直接输出控制量（归一化的线速度和转向角）。

### 动作空间 (2维)

| 维度 | 含义 | 归一化范围 | 实际范围 | 说明 |
|------|------|------------|----------|------|
| 0 | 线速度 | [-1, 1] | [0, 3] m/s | 只允许前进时 |
| 1 | 转向角 | [-1, 1] | [-1, 1] rad | 左负右正 |

### 训练命令

```bash
# 基础训练（默认只允许前进）
python train_control.py --total-timesteps 1000000

# 带实时可视化训练
python train_control.py --total-timesteps 1000000 --visualize

# 调整渲染速度
python train_control.py --visualize --render-delay 0.02

# 允许车辆后退
python train_control.py --total-timesteps 1000000 --no-forward-only
```

### 可视化显示内容

- 车辆运动轨迹
- **全局参考路径（蓝色虚线）** - 从起点到终点
- 激光雷达扫描范围
- 障碍物位置
- 目标位置（红色星号）

---

## 方法二：规划型 (planning_gym)

模型输出目标加速度和目标转角，通过运动学模型预测轨迹，MPC跟踪器计算实际控制量。

### 动作空间 (2维)

| 维度 | 含义 | 归一化范围 | 实际范围 | 说明 |
|------|------|------------|----------|------|
| 0 | 加速度 | [-1, 1] | [-3, 3] m/s² | 加速/减速 |
| 1 | 转向角 | [-1, 1] | [-1, 1] rad | 左负右正 |

### 工作流程

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│  模型输出    │ ──▶ │  运动学模型预测   │ ──▶ │  MPC跟踪器  │
│ [加速度,转角]│     │  未来horizon步轨迹 │     │  计算控制量  │
└─────────────┘     └──────────────────┘     └─────────────┘
                                                   │
                                                   ▼
                                            [速度, 转角]
```

### 训练命令

```bash
# 基础训练（默认只允许前进）
MPLBACKEND=Agg python train_planning.py --total-timesteps 1000000

# 多进程并行训练（推荐，速度约 3x）
MPLBACKEND=Agg python train_planning.py --total-timesteps 1000000 --n-envs 4 --batch-size 256 --learning-starts 10000

# 更多并行（CPU核数足够时）
MPLBACKEND=Agg python train_planning.py --total-timesteps 1000000 --n-envs 8 --batch-size 256 --buffer-size 1000000

# 带实时可视化训练（单环境）
python train_planning.py --total-timesteps 1000000 --visualize

# 调整渲染速度
python train_planning.py --visualize --render-delay 0.02

# 允许车辆后退
MPLBACKEND=Agg python train_planning.py --total-timesteps 1000000 --no-forward-only
```

> **说明**：`--n-envs > 1` 时自动使用 `SubprocVecEnv` 多进程并行采样；`--visualize` 模式强制单环境运行。无显示器时需加 `MPLBACKEND=Agg` 前缀。

### 可视化显示内容

| 内容 | 显示方式 | 说明 |
|------|----------|------|
| 车辆运动 | 实时更新 | 当前车辆位置和航向 |
| **全局路径** | **蓝色虚线** | 从起点到终点的参考路径 |
| **规划轨迹** | **绿色虚线** | 运动学模型预测的轨迹 |
| 目标点 | 红色星号 | 最终目标位置 |
| 激光雷达 | 红色射线 | 障碍物检测 |
| 边界墙壁 | 灰色矩形 | 环境边界 |

---

## 两种方法的对比

### 动作空间对比

| 特性 | 控制型 | 规划型 |
|------|--------|--------|
| 维度0 | 速度 [-1, 1] → [0, 3] m/s | 加速度 [-1, 1] → [-3, 3] m/s² |
| 维度1 | 转向角 [-1, 1] → [-1, 1] rad | 转向角 [-1, 1] → [-1, 1] rad |

### 架构对比

| 特性 | 控制型 | 规划型 |
|------|--------|--------|
| 决策层级 | 低层控制 | 中层规划 |
| 输出直接性 | 直接控制车辆 | 通过MPC间接控制 |
| 安全性 | 依赖策略学习 | MPC提供安全约束 |
| 训练难度 | 较低 | 较高 |
| 可解释性 | 较低 | 较高（有预测轨迹） |

### 可视化区别

```
控制型可视化:
┌─────────────────────────┐
│  🚗 车辆                │
│  ││ 激光雷达            │
│  ● 障碍物               │
│  ★ 目标（红色星号）      │
│  - - - 全局路径（蓝色）  │
└─────────────────────────┘

规划型可视化（多显示规划轨迹）:
┌─────────────────────────┐
│  🚗 车辆                │
│  ││ 激光雷达            │
│  ● 障碍物               │
│  ★ 目标（红色星号）      │
│  - - - 全局路径（蓝色）  │
│  · · · 规划轨迹（绿色）  │  ← 规划型特有！
└─────────────────────────┘
```

---

## 参数说明

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--total-timesteps` | 1000000 | 总训练步数 |
| `--learning-rate` | 3e-4 | 学习率 |
| `--batch-size` | 256 | 批次大小 |
| `--buffer-size` | 1000000 | 经验回放池大小 |
| `--learning-starts` | 10000 | 开始训练前的随机探索步数 |
| `--train-freq` | 1 | 每隔多少步做一次更新 |
| `--gradient-steps` | 1 | 每次更新执行的梯度步数 |
| `--gamma` | 0.99 | 折扣因子 |
| `--tau` | 0.005 | 目标网络软更新系数 |
| `--policy-delay` | 2 | Actor 延迟更新频率 |
| `--action-noise-std` | 0.1 | 探索噪声标准差 |
| `--device` | auto | 设备 (auto/cpu/cuda) |
| `--save-dir` | ./logs | 保存目录 |
| `--eval-freq` | 5000 | 评估频率 |

### 可视化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--visualize` | False | 启用实时可视化（每步渲染） |
| `--render-delay` | 0.01 | 渲染帧间隔(秒) |

### 车辆运动参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--forward-only` | True | 只允许车辆前进（默认开启） |
| `--no-forward-only` | False | 允许车辆后退 |

---

## 评估模型

```bash
# 控制型评估
python eval_control.py --model-path ./logs/control_ppo_xxx/final_model.zip --visualize

# 规划型评估
python eval_planning.py --model-path ./logs/planning_td3_xxx/final_model.zip --visualize
```

---

## 训练指标

训练过程中会自动跟踪以下指标：

| 指标 | 含义 | 说明 |
|------|------|------|
| `success_rate` | 成功率 | 到达终点的回合比例 |
| `collision_rate` | 碰撞率 | 发生碰撞的回合比例 |
| `timeout_rate` | 超时率 | 超过最大步数的回合比例 |
| `total_episodes` | 总回合数 | 已完成的回合数量 |
| `ep_len_mean` | 平均回合长度 | 平均每回合步数 |
| `ep_rew_mean` | 平均回合奖励 | 平均每回合总奖励 |

### 指标含义

- **成功**: 车辆到达终点（`arrive_flag == True`）
- **碰撞**: 车辆与障碍物或边界碰撞（`collision_flag == True`）
- **超时**: 回合步数超过 `max_steps` 限制

### 示例输出

```
| rollout/           |          |
|    collision_rate  | 0.85     |  <- 85% 的回合发生碰撞
|    ep_len_mean     | 136      |  <- 平均每回合 136 步
|    ep_rew_mean     | -74.2    |  <- 平均奖励 -74.2
|    success_rate    | 0.05     |  <- 5% 的回合成功到达终点
|    timeout_rate    | 0.10     |  <- 10% 的回合超时
|    total_episodes  | 15       |  <- 已完成 15 个回合
```

---

## 监控训练

```bash
tensorboard --logdir ./logs
```

浏览器访问 http://localhost:6006

---

## 常见问题

### 1. 可视化窗口不显示

确保安装了交互式matplotlib后端：
```bash
pip install tk
```

### 2. 可视化训练太慢

- 增大 `--render-delay` 参数
- 或先无可视化训练，再单独可视化评估

### 3. 导入错误

如果遇到导入错误，请确保：
1. 激活了虚拟环境：`source .venv/bin/activate`
2. 在 workspace 目录下运行脚本
3. 已完成安装步骤（见文档顶部）

### 4. 碰撞过多

这是正常的，初始策略是随机的。随着训练进行，智能体会学会避障。

---

## 修改环境配置

编辑 `env.yaml` 文件可以自定义环境：

```yaml
# 修改世界大小
world:
  height: 20    # 高度 (m)
  width: 37     # 宽度 (m)

# 修改起点和终点
robot:
  - state: [2, 10, 0, 0]     # 起点 [x, y, theta, v]
    goal: [35, 10, 0]         # 终点 [x, y, theta]
    vel_max: [3, 1]           # 最大速度 [线速度, 角速度]

# 修改障碍物
obstacle:
  - state: [[10, 7], [15, 10], ...]  # 障碍物位置
    shape:
      - {name: 'circle', radius: 1.2}
```
