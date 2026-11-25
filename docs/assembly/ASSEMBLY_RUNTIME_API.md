# 装配在线证据接口

`scene_factory.assembly_runtime.AssemblyRuntimeEvaluator` 是纯 Python 判据模块。它消费仿真测量，不写入位姿，也不执行控制。适配器必须在 episode 开始调用 `reset(episode_id)`，随后每个物理步调用 `record(...)`。读取现有结果用 `status()`；结束未完成轨迹用 `finalize()`；导出证据用 `trace()`。

```python
evaluator = AssemblyRuntimeEvaluator(spec, physics_dt_s=dt)
evaluator.reset(episode_id)
status = evaluator.record(
    sim_time=sim_time,
    objects={
        "peg_1": {"position_m": p, "orientation_wxyz": q,
                  "linear_velocity_mps": v, "angular_velocity_rps": w},
        "socket_1": {"position_m": p2, "orientation_wxyz": q2,
                     "linear_velocity_mps": v2, "angular_velocity_rps": w2},
    },
    contacts=[{"pair": ["peg_1", "socket_1"], "normal_force_N": force_sum}],
    coverage={"complete": True, "continuous": True,
              "scope": "all_contacts_involving_task_parts",
              "object_ids": ["peg_1", "socket_1"],
              "force_aggregation": "sum_per_point_normal_magnitudes",
              "sample_dt_s": dt, "source": "backend measurement source"},
    action={"mode": "joint_position", "payload": joint_targets},
    released=False,
    geometry_clearance_m=clearance,
)
```

`contacts` 需要覆盖两个任务零件的所有接触，包括零件与机器人、桌面及其他物体。每个条目可是一接触点，也可是一对象对所有接触点的法向力大小之和；同一对象对的条目会相加。不能用合力向量的模代替逐点法向力大小之和。可选 `normal_force_points_N` 会核验其和。`coverage` 是后端测量声明，并非该纯模块对传感器完整性的独立证明。

缺失 SE(3)、实时速度、接触覆盖或连续物理步会将本 episode 标为 `not_verified`，后续目标位姿不能消除缺失证据。`reset()` 才能清除旧证据。无新控制命令但物理继续推进时传 `action=None`；实际施加的关节目标可在每步重复记录。`released` 应来自真实夹爪释放控制及接触/夹持状态，不能以最终位姿推测。

逐步轻量检查会立即处理超力、未允许接触、直接写位姿、掉落和实时速度越界。完整 frozen `AssemblySpec` 轨迹判据默认每 0.1 仿真秒重新计算一次，所有物理步仍完整保存；可用 `evaluation_interval_s=0` 每步完整评估。奖励仅在完整保持与释放稳定窗口通过后发出一次。`status()` 不重复发奖励。完整轨迹含正负例，可另用离线判据核验；本接口本身不代表任何仿真后端或机器人已完成装配。
