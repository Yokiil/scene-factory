# MuJoCo 运行、重置与状态恢复

日期：2026-10-28。本页记录本地 CPU 后端已实现和实测的能力，不能据此推断 Isaac
接触精度、机器人装配成功率或 Linux GPU 性能。

## 模型复用

`MujocoBackend(reuse_model=True)` 默认复用当前后端的已编译模型；
`reuse_model=False` 保留每次重新编译的对照路径。缓存只有一个当前模型，不跨会话共享可变
`MjModel`/`MjData`。完整的实际导出 XML 参与比较；资产质量、惯量、摩擦、碰撞几何、求解器、
静态位姿和结构的变化都会触发重新编译。

直接挂在 world 下、没有子 body 的 free body 允许只改变初始平移和旋转：其他 XML 必须
一致，然后更新模型的初始自由关节状态和参考位姿，重算常量并执行 `mj_resetData`。
关节速度、力、控制、暖启动状态及任务计数器恢复初值。更复杂的拓扑或嵌套自由体不走这条路径。
每次 reset 重新生成实际 XML，避免仅凭场景 ID 或旧资产路径误复用。

reset 的 `info` 返回 `reset_kind`、`model_sha256`、`model_compile_count`、
`model_cache_hits` 和 `model_reset_count`。模型摘要对应本次实际 XML，即使复用了编译结构，
不同初始位姿仍有不同的配置摘要。模型复用不允许复用旧物理/任务验收证据。

## 实例物理参数

动态对象 JSON 可指定：

```json
{
  "physics": {
    "mass_kg": 1.0,
    "inertia_kg_m2": [0.001, 0.001, 0.001],
    "friction": 0.5,
    "source": "constrained_sample"
  }
}
```

必须同时给出这些字段。质量和惯量须为有限正数，主惯量须满足严格三角不等式；摩擦
须为有限非负值，允许零摩擦。未知字段拒绝。当前只支持动态 primitive 碰撞代理，
外部网格的实例参数覆盖会明确报错。网格缩放不能只改 `bbox_m`；随机化调用方应拒绝
未接入缩放的网格资产。原资产注册表保持不变。

## 物理观测

旧的 `object_poses` 和 `proprioception.qpos/qvel` 保留。新增：

- `objects[id]`：`position_m`、`orientation_wxyz`、`linear_velocity_mps`、
  `angular_velocity_rps`、质心坐标；速度按 MuJoCo body 中心处、世界轴表达。
- `contacts`：逐接触的两个对象和 geom、接触位置、间距、接触坐标轴、接触系六维力/力矩，
  以及对 geom2 作用的世界系力和力矩。力单位 N，力矩单位 N·m。
- `sim_time`：MuJoCo `data.time`。每个环境步后执行 forward，使位姿、速度和接触
  对应同一状态时间。接触值类型为 `instantaneous_solver_force`，采样时间窗为 0，
  不是上一控制周期的平均力或冲量。

对象的 `scene_graph` 仍为初始结构，实际动力学位姿应读取 `objects`。
MuJoCo 广义坐标不是机器人关节接口，此后端没有因这些新增字段而获得 Franka 控制能力。

## 运行状态恢复

```python
snapshot = backend.capture_state()
# 执行动作……
observation, info = backend.restore_state(snapshot)
```

`scene_factory.mujoco_state.v1` 使用 MuJoCo `mjSTATE_INTEGRATION` 保存时间、qpos/qvel、
内部激活、暖启动、控制、外力、约束开关、mocap、userdata 和引擎支持的插件状态。
同时保存环境步数、完整任务计数器、任务初始状态及最后一次任务判定。

恢复要求实际模型和场景摘要、MuJoCo 版本、后端实现源文件摘要、frame skip/max steps
完全匹配。校验和、有限数值、数组大小、单位四元数、时间/步数一致性和任务状态在提交前检查。
候选 `MjData` 和任务状态在独立对象中构造，失败时原活动状态保持完整。
不通过调用任务判定推进计数器来“重建”保存的任务状态。

这是同后端兼容版本的运行状态恢复，不是跨引擎/跨硬件承诺。接触派生量由 forward 重算；
渲染器、外部 RNG 和未接入的传感器状态不在快照中。本后端 step 本身不使用随机数。
精度以目标运行环境上的续跑对照为准，不把本次零误差扩大到其他平台。

## 本机测量

使用 `tools/validate_mujoco_state.py` 对相同的 6 物体客厅场景交替测量两个路径，
各 100 次 reset；不渲染，包含状态和接触观测。

| 路径 | P50 | P95 |
| --- | ---: | ---: |
| 每次重新编译 | 2.143 ms | 3.225 ms |
| 模型复用 | 0.663 ms | 0.943 ms |

最终复测中位耗时下降约 69.1%；同一进程、Windows、MuJoCo 3.3.7，开发机后台负载未隔离。
这不是冷启动或跨 GPU 的性能结果。

后续 1000 次“施力一步→重置”测试，完整 integration state 与初值的最大差异为 0，
模型仅编译一次。采样的进程 private bytes 保持不变，工作集增加 102400 字节；
这是有限时长下的观测，不能证明所有运行负载绝无泄漏。
同一模型恢复后续跑 100 个环境步，qpos/qvel 的最大绝对差异均为 0。

最终数据在 `outputs/delivery_20261028/mujoco_reset_restore_validation_final.json`，包含全部耗时、
内存样本、完整场景、环境信息和当前任务判定实现的源码摘要。较早一轮证据保留为
`mujoco_reset_restore_validation.json`，未覆盖。复测：

```sh
python tools/validate_mujoco_state.py --output outputs/mujoco_reset_restore_validation.json
```

自动化测试另覆盖：直接 free body 位姿随机化与全量重编译的 30 步等价性、接触状态续跑、
非法快照原子拒绝、任务计数器恢复、几何和物理参数变化失效，以及施力请求校验无副作用。

## 本机 HTTP 服务基准

`tools/benchmark_runtime_service.py` 通过公开 RuntimeClient 向真实 loopback HTTP 服务发请求。
管理进程常驻，OS 文件缓存未清空，资产已在本地；每次创建启动一个新 worker。
没有空闲 worker 池，因此不把这组结果写成“温会话创建”，也没有测未缓存主机冷启动。

| 操作 | 样本数 | P50 | P95 |
| --- | ---: | ---: | ---: |
| 新 worker 创建并 reset 到首个状态观测 | 30 | 708.553 ms | 772.572 ms |
| 同 worker、同拓扑、同 seed reset | 100 | 20.537 ms | 32.551 ms |
| 一次 HTTP step、一个物理步 | 100 | 14.851 ms | 33.417 ms |

全部请求成功，创建使用了 30 个不同 worker PID。场景为 6 个碰撞代理（3 静态、3 动态），
dt=0.002 s、action repeat=1，Newton solver、100 iterations、implicitfast；状态含接触，
不渲染。耗时包含 HTTP 编解码、worker IPC 和 SQLite 证据写入；不含服务启动、场景构建、
资源释放及报告写盘。它衡量串行客户端交互延迟，不代表引擎内极限吞吐。

运行环境：Windows 11、Python 3.12.7、MuJoCo 3.3.7、i9-13980HX；RTX 4060 Laptop
GPU 仅记录在硬件清单，仿真没有使用 GPU。测量前已协调停止其他 SceneFactory 测试，
操作系统后台进程仍未隔离。报告含原始耗时、接触数量、场景/源码摘要以及失败计数：
`outputs/delivery_20261028/http_service_benchmark_final.json`。

```sh
python tools/benchmark_runtime_service.py --output outputs/http_service_benchmark.json --creation-trials 30 --reset-trials 100 --step-trials 100
```

该脚本拒绝覆盖已有报告。HTTP 基准与上表直接后端 reset 基准是不同统计边界，不能把两者
当作同一优化前后的数据；本轮也未验收 Linux、远程网络或 Isaac 高精度接触吞吐。
