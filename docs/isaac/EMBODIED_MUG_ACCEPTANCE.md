# Franka Mug 闭环操作与物理验收

本轮以 main `524019cdd49f32be9c65257387b08b654565dd77` 为基线。固定 `kitchen_franka_mug_pick_place`、seed 77、官方 Franka Panda 和 ready 的 YCB `mug_001`。机器人在岛台源区夹起杯子，移动到目标矩形，再张爪将杯子放回岛台。所有动作在本机 Isaac Sim 6.0.1.0 / PhysX 中执行。

## 2026-11-27 实测结果

三次正式运行读取相同的 Layout 和 USD，分别保存进程退出码 0、原始动作和逐物理步观测。回放逐帧重算了 TaskEvaluator。下表数字为物理步，从初始观测 0 开始，每步 1/60 s。接触力是 PhysX 测得的机器人与杯子全部接触点法向力幅值之和。

| 控制方式 | 抓取（抬升 ≥ 10 cm） | 持杯进入目标 XY | 张爪脱离 | 稳定放置终步 | 最后连续稳定步数 | 机器人与杯子峰值 |
|---|---:|---:|---:|---:|---:|---:|
| 原 scripted | 338 | 422 | 479 | 615 | 121 | 50.37 N |
| 闭环 Agent | 331 | 415 | 473 | 608 | 136 | 26.98 N |
| 闭环 Agent，首抓偏移 | 918 | 1002 | 1060 | 1195 | 109 | 26.75 N |

恢复回合第 361 步实际空夹，微抬验收返回 `grasp_failure`。第 362 步开始开爪，第 414 步进入垂直退避，第 426 步开始驱动关节回中，第 588 步以新测得的杯位开始第二次尝试。原始高度基准、任务判定器、episode 身份与物理步数贯穿全程。最终杯中心 `(0.6182, 0.1537, 0.9638) m`。正常成功回合的最终杯中心为 `(0.6180, 0.1549, 0.9638) m`。正常与恢复回合均经过独立进程复跑。

| 验收层次 | 结果 | 范围 |
|---|---|---|
| 代码 | 通过 | 本轮相关回归共 232 项测试和 20 个子测试通过；核心改动通过 Ruff 检查 |
| 接口 | 通过 | 本地真实 MuJoCo HTTP 会话查询 WorldState 时，3 次物理步使版本从 1 变为 4，时钟到达 0.03 s；重复读取返回同一帧 |
| 物理任务 | 通过 | 官方 Franka / YCB 杯，原 scripted 对照、注册技能正常回合与同一回合失抓恢复均通过 |
| 策略泛化 | 单样本通过，整体待验证 | 固定回合通过后，用 `--random-seed 78` 运行已有随机杯位配方。该次初始实测坐标 `(0.5638, 0.0093, 0.9644) m`，第 607 步通过。样本范围限于同一杯、同一机器人和岛台 |

证据包中 `verified_scripted`、`verified_agent`、`verified_recovery` 为正式结果，`probe_seed78` 为随机样本。其他目录保留开发期间的实际失败、被中断的启动及夹爪校准尝试，包括超力停止、滑脱与退避超时。`episode.json` 记录正式运行的代码和输入哈希，`acceptance.json` 记录原始动作及观测 SHA-256。策略使用本地规则和注册技能，检测依赖仿真实体身份及 PhysX 位姿，JEV 推理与视觉感知未纳入本次验收。

## 接口审计与复用

| 原模块 | 已有能力 | 本轮接入边界 |
|---|---|---|
| SceneSpec / Layout / Scene Graph | 配方、已求解的实例和初始空间关系 | 保留初始位姿和资产声明，单列当前实测空间关系 |
| SceneFactoryEnv / SimulatorBackend | reset、step、render 及后端能力声明 | 同帧观察投影到 WorldState，读取操作不推进仿真 |
| RuntimeClient / HTTP Session | episode 身份、state_version、幂等 step、缓存观测、快照 | 新增 `client.world_state(session_id)` 和 `GET /v1/sessions/{id}/world-state`，复用会话锁和已提交帧 |
| IsaacSimBackend / MuJoCo | Isaac 已有刚体 tensor、接触力和 Franka IK。MuJoCo 有 MjData 和刚体接触 | 两者都能提供 WorldState。此任务使用 Isaac 手臂，MuJoCo 刚体外力不作为机器人操作 |
| RobotSpec / RobotAdapter | Franka 关节名、指爪规格、运动学坐标系及适配协议 | 复用后端的 Lula 和关节驱动，每个技能调用仅推进一个物理步 |
| InteractionExecutor | 干运行交互计划与最终关节状态检查 | 保留规划用途。抓杯执行直接使用已有 Isaac 物理控制器 |
| agent_adapter / JevDecisionPolicy | 注册固定动作、证据前置条件、模型候选技能筛选 | 本演示为本地观测驱动策略。无模型服务调用 |
| TaskEvaluator / AssemblyRuntimeEvaluator | pick-and-place 已有抓持、释放、区域和稳定窗。装配另有接触历史判据 | 在线使用原 pick-and-place evaluator，离线从原始力和位姿再算一次。装配判据保持其原有任务范围 |

## 复现

使用安装 Isaac 的 Python 3.12，在仓库根目录依次运行。`$base` 取新的 ASCII 路径，每个目录只接受一次写入。三个进程独立加载场景，各模式使用相同的物理设置。`recovery` 首次将夹爪目标相对正常候选平移 `(-0.15, -0.16, +0.005) m`，实际执行一次空夹。恢复策略要求真实失抓、当前指杯脱离、杯子连续稳定和源支撑正向力。PhysX 会省略休眠体的接触力，程序在微抬确认的最后四步及恢复阶段调用原生 `wake_up`，再从下一物理步读取接触力并记录唤醒动作。夹爪张开后，机械臂垂直退避并通过关节驱动回中，用新测得的杯子位姿重抓。

```powershell
$python = '<ISAAC_PYTHON>'
$assets = '<OFFICIAL_LOCAL_ASSETS>'
$base = '<ASCII_RUNTIME_DIR>\repeat_mug_01'
foreach ($mode in 'scripted','agent','recovery') {
    & $python tools/run_embodied_mug.py --asset-root $assets --mode $mode --output "$base\$mode"
    if ($LASTEXITCODE -ne 0) { throw "Episode failed: $mode" }
}
```

每回合分别写入原始观测 `trajectory.jsonl` 和执行前的请求 `actions.jsonl`。第 *k* 条动作引用 `from_step=k`，其结果在 `simulation_step=k+1` 的观测中。该帧同时含实际笛卡尔命令、关节目标、实测 TCP、杯位姿与速度、指爪接触力、全场景接触以及原任务 oracle。`worker.json` 包含技能决策和失败历史，`acceptance.json` 包含独立回放的检查结果及各阶段首次见证。崩溃或中途安全停止均保留已写入的日志。

### WorldState

Python 的 `build_world_state(scene, observation, *, backend, episode_id, step_index, ...)` 和 HTTP 返回 `scene_factory.world_state.v1`。`timestamp` 取模拟时钟秒。`objects[id].instance` 记录场景初始条件，`asset_metadata` 描述资产交互声明，`position_m`、`orientation_wxyz`、两项速度和 `joint_states` 表示当前测量。`robot` 的关节和末端状态来自 articulation。`spatial_relations` 区分空间距离和接触，`affordances` 标记来源及当前可用性，`task_progress` 与 `task_oracle` 表示任务进展。后端没有测量的字段使用 `null`。此单场景的检测通过仿真实体 ID 和刚体数据完成。

## 验收口径

抓取要求实际指杯接触且杯子相对于同回合初始实测位姿上升至少 0.10 m。搬运见证要求仍然抓持、杯子悬空且进入目标 XY 区。释放见证检查已张爪、有效力读数和指杯脱离。最终放置要求目标区、支撑体接触、释放、原 0.03 m 高度容差及至少连续 20 步位移不超过 0.005 m，同时通过原控制器和 TaskEvaluator。离线回放检查逐帧时间连续性、接触与 oracle 一致性、阶段转移和官方 Franka 资产。恢复回合还检查同一连续物理轨迹上的失抓、真实开爪退避和第二次瞄准。
