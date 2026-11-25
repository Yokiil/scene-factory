# 物理沙盒首阶段实现

> 历史阶段记录。后续已补充 HTTP 服务、外部动作、图像与恢复能力；当前交付状态及未完成项见
> [2026-10-28 交付说明](SANDBOX_DELIVERY_20261028.md)。下文“当前”指首阶段完成时。

日期：2026-10-28。本阶段交付运行基准、版本化会话契约和有效的 episode 重置。
这是现有后端上的最小运行接口，不是整套优化方案的验收完成声明。

## 已实现范围

| 能力 | 当前实现 |
| --- | --- |
| 会话 | Python `RuntimeSession`，串行 reset/step/observe/close，CLI smoke 共用此核心 |
| 动作请求 | episode ID、从 0 开始的请求序号、幂等键、动作重复次数和能力检查 |
| 观测 | 状态版本、episode 场景摘要、步数、可用时的仿真时间、原始状态、奖励及终止信息 |
| reset | 动态叶子物体 XY/yaw 初始位姿覆盖与随机扰动，实际 seed、采样值及配置摘要 |
| 基准 | 创建后端、首次 reset、重复 reset、step、可选 render、close、写盘的原始样本及 P50/P95 |
| 可追溯性 | 基准记录场景/registry/代码摘要、引擎/求解器/主机信息；检查测试期间输入未变化 |

`schemas/runtime_session.schema.json` 与 `runtime_session_schema()` 保持一致；Schema
描述插入默认值后的消息。运行时校验还会检查后端能力、对象存在性和动态属性。
已有 `.scene.zip` v1、布局格式和语义 `TaskGraph` 的含义不变。

### 后端能力边界

| 后端 | 会话支持的动作 | 初始位姿选项 | 证据范围 |
| --- | --- | --- | --- |
| `dry_run` | `noop`、`dry_run` | 支持 | 仅协议和数据流，无物理 |
| `mujoco` | `noop`、`object_forces` | 支持 | 当前 MJCF 碰撞代理；没有机器人控制器 |
| `isaac` | `scripted` | 拒绝会改变场景的选项 | 现有 Franka 抓杯/搬放脚本切片 |

CLI 将 `dry-run` 映射为契约中的 `dry_run`。`object_forces` 为世界坐标下作用于
物体质心的力 `[Fx,Fy,Fz]`（N），可附力矩 `[Tx,Ty,Tz]`（N·m）；同一请求重复执行时
每一步重新施加。它不是机器人关节位置目标。所有目标均在步进前校验，静态物体和未知 ID 拒绝。

当前会话观测只支持状态；图像引用为空，缺失的姿态、速度、接触等字段明确列入
`unavailable_fields`。原始 `scene_graph` 为初始化结构，不代表当前动力学位姿。
其中 `pose.yaw_deg` 单位仍为度；关节角采用弧度，四元数约定为 wxyz。
MuJoCo `qpos/qvel` 是场景广义坐标，不能当作 Franka 关节状态。
`sim_time` 是 episode 内的相对时间，dry-run 返回 null。`observe()` 读取最后一次已提交
观测，不渲染、不推进仿真；不声明 RGB-D 与状态对齐已完成。

## 使用方式

在项目 Python 3.12 环境中执行；Windows 可将下面的 `python` 换为
`.venv\Scripts\python.exe`。MuJoCo 命令需要已安装对应可选依赖。

```sh
python -m scene_factory.cli sandbox capabilities
python -m scene_factory.cli sandbox schema
python -m scene_factory.cli build --recipe living_room_returned_home --seed 19 --output outputs/sandbox_scene
python -m scene_factory.cli sandbox smoke outputs/sandbox_scene/layout.json --backend mujoco --steps 5 --output outputs/session_smoke.json
python -m scene_factory.cli sandbox benchmark outputs/sandbox_scene/layout.json --backend mujoco --creation-trials 30 --reset-trials 100 --step-trials 100 --output outputs/mujoco_baseline.json
```

`sandbox smoke` 在一个进程内创建、重置、步进并释放会话，同时验证重试回执一致。
可用 `--reset-options options.json`、`--action action.json` 传入配置。
失败报告包含异常类型与原因，退出码为 1；成功仅表示协议冒烟通过。
`sandbox benchmark --render` 单独记录采样图像的开销；不支持图像的后端会明确失败。
Isaac 命令须在 Isaac Python 中执行并指定 `--usd`；Windows 根 USD 路径须符合现有 ASCII 限制。

Python 接口示例：

```python
import json
from pathlib import Path

from scene_factory.agent import SceneFactoryEnv
from scene_factory.backends.mujoco import MujocoBackend
from scene_factory.session import RuntimeSession

path = Path("outputs/sandbox_scene/layout.json")
scene = json.loads(path.read_text(encoding="utf-8"))
parents = {o.get("support", "").partition(":")[0] for o in scene["objects"]
           if isinstance(o.get("support"), str)}
target = next(o["object_id"] for o in scene["objects"]
              if o["dynamic"] and o["object_id"] not in parents)

with RuntimeSession(SceneFactoryEnv(path, MujocoBackend())) as session:
    initial = session.reset({
        "idempotency_key": "reset-1", "seed": 42,
        "options": {"pose_jitter": {target: {"x_m": [-0.001, 0.001]}}},
    })
    request = {
        "episode_id": initial["episode_id"], "sequence_id": 0,
        "idempotency_key": "action-1", "steps": 5,
        "action": {"mode": "noop", "payload": {}},
    }
    result = session.step(request)
    assert session.step(request) == result  # 不再次执行 5 步
    assert session.observe() == result
```

### 重置规则

`SceneFactoryEnv.reset(seed=..., options=...)` 仍返回 `(observation, info)`，保持旧后端
`reset(scene)` 适配方式。`info.episode_reset` 包含实际 seed、RNG 算法标识、配置、
源场景/episode 场景摘要、采样偏移、最终位姿和验证范围。

```json
{
  "pose_overrides": {"dynamic_leaf_id": {"position_xy_m": [0.1, 0.2], "yaw_deg": 15}},
  "pose_jitter": {"dynamic_leaf_id": {"x_m": [-0.001, 0.001], "yaw_deg": [-1, 1]}}
}
```

`dynamic_leaf_id` 应替换为场景中的动态叶子 ID。先应用绝对覆盖，再加均匀随机偏移；
高度不变。支持其他物体的父对象不得修改。每个对象/字段有独立随机数流，新增无关随机变量
不会改变已有字段的抽样。seed 范围为 `[0, 2**64)`；省略时生成并记录 seed，可将其传回复现。
只指定 seed 而没有随机变量时，场景保持原值，并记录 `randomization_applied=false`。

采样不修改布局 seed、源文件或共享资产。拒绝未知选项、无效范围及非有限数值。
这些检查仅证明初始化契约有效，不检查碰撞、支撑稳定性、可达性或任务可解性。
实例级质量/摩擦采样和受物理约束的拒绝采样尚未实现。Isaac 使用预导出的 USD，
因此场景变化被拒绝，避免 JSON 与实际仿真对象不一致。

### 请求和故障规则

- `sequence_id` 对每个 step 请求递增一次；`step_index` 按实际执行的环境步递增。
  动作重复遇到 terminated/truncated 会提前停止，奖励为已执行步的总和。
- 同一 episode 的相同键和内容返回原回执；同键不同内容、乱序或旧 episode 请求拒绝。
  已结束 episode 可重取成功回执，但不能执行新的动作。
- 后端可能已经部分执行再抛异常，故此时会话进入 faulted，禁止继续步进或读取旧状态，
  必须用新幂等键成功 reset。旧 reset 回执不能当作恢复成功。
- 后端开始 reset 前的输入校验失败不污染有效会话。reset 失败后关闭后端并使场景摘要失效。
- 重试保护仅存于当前进程对象中，不跨崩溃/重启持久化。默认保存至多 4096 个 reset 键，
  每个 episode 至多 4096 个 step 键；到限额时拒绝新操作，不驱逐旧键造成重复执行。
  reset 历史满后重建会话，step 历史满后重置 episode。
- `RuntimeSession` 独占其环境；使用会话期间不要绕过它直接调用 backend 或 Env。

## 基准解释

`in_process_creation` 为同一个 Python 进程中构造全新后端到首个状态观测的耗时，
包含首次 reset。`reset` 为同一 Env 对象、同一输入和 seed 的重复重置。
MuJoCo 当前仍每次重建模型，不能把此项称为热重置优化结果。
编译/引擎启动/场景加载目前包含在后端 reset 内，报告明确标注未进一步分解。

报告保留每次耗时、失败样本和 P50/P95（成功样本线性插值），并注明样本量。
吞吐仅按成功 step 调用耗时计算，不含 reset、单独渲染和网络耗时。
场景、registry、已识别的 MuJoCo 外部网格在运行期间发生变化会使报告失败；
USD 外部依赖闭包尚未锁定，报告不会把根 USD 的哈希当作完整环境包证明。

本机实测与原始报告见 [首轮基准记录](SANDBOX_BASELINE_20261028.md)。
没有测量的指标保持 `unmeasured`，包括真正进程冷启动、常驻 worker 温启动、峰值资源、
1000 次重置污染/泄漏、恢复误差和 Linux GPU 部署。不能用本轮 CPU 代理场景吞吐推断 Isaac。

## 下一验收门槛

下一步应先接入 Isaac 对齐图像/状态、首种外部关节动作，并在固定 Linux/NVIDIA/Isaac
环境完成服务器抓杯 smoke。持久 HTTP 会话、worker 池、热重置、任意状态恢复仍未实现。
方案中的 `/v1/sessions` 等端点及跨 CLI 调用的会话生命周期不在本次交付中。
环境包锁文件设计见 [RUNTIME_LOCK_DESIGN.md](RUNTIME_LOCK_DESIGN.md)，当前只完成设计，未强制执行。
