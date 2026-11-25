# 环境编译、物理采样与装配契约交付

日期：2026-10-28。以下代码和本地证据已实现；每部分注明其验收边界。

## 环境包与锁文件

`scene_factory.environment_package` 提供：

```python
from scene_factory.environment_package import build_environment, verify_environment, archive_environment

manifest = build_environment(
    "layout.json", "environment", registry_path="data/assets/registry.jsonl",
    backend="mujoco", cache_dir="environment_cache",
)
assert verify_environment("environment")["valid"]
archive_environment("environment", "environment.zip")
```

输出目录必须尚不存在，避免覆盖已有交付物。新格式 `scene_factory.environment_package.v1`
独立于已有 `.scene.zip` v1。目录布局：

对应 CLI：

```sh
python -m scene_factory.cli sandbox build layout.json --output environment --cache environment_cache
python -m scene_factory.cli sandbox verify environment
python -m scene_factory.cli sandbox archive environment --output environment.zip
python -m scene_factory.cli sandbox randomize layout.json randomization.json --seed 0 --count 1000 --output candidates
python -m scene_factory.cli sandbox dependency-diff before.json after.json --output invalidation.json
python -m scene_factory.cli sandbox assembly-evaluate assembly_spec.json trace.json --output assembly_result.json
python -m scene_factory.cli sandbox room-benchmark --output room_results --physics-steps 1000
```

`randomize --count 1` 写出实际场景和采样记录；批量模式保存源场景、配置和每个候选的实际值，
可按记录 seed 复现。`verify --content-only` 明确跳过本地运行时兼容校验。
未通过参数约束、装配判据、包校验或房间静置判据时，CLI 返回退出码 2。

```text
manifest.json
scene/layout.json
scene/scene.xml
assets/registry.jsonl
assets/source/<selected mesh asset>/SOURCE.json
assets/source/<selected mesh asset>/geometry.json
runtime/dependencies.json
runtime/lock.json
```

构建冻结当前 MuJoCo 导出器使用的完整输入：所选资产注册记录及其读取的外部网格。
生成的 MJCF 几何全部内嵌；若导出器引入未处理的外部文件引用，构建直接失败。
注册表中的未使用 USD/GLB 来源字段仅作为元数据，不表示这些资源属于当前代理后端的依赖。
资产缺少声明的许可或必要文件时失败。

锁文件记录每个依赖的 SHA-256/大小、引擎与 Python 精确版本、导出器/模型/后端代码摘要、
任务、实例实际编写的物理参数及求解器设置。由 MuJoCo 根据锁定形状推导的惯量明确标为
`mujoco_compiler_from_locked_geometry`，没有伪装成实测值。默认验证同时检查本地运行时
版本与源码兼容，并从冻结输入重新生成 MJCF 比对；`check_runtime=False` 仅做包内容校验。
环境摘要不含缓存命中统计，不会因第二次构建而改变。

当前只支持 MuJoCo 完整包。Isaac 的 USD/纹理/机器人/插件闭包尚未冻结，选择 Isaac 会明确拒绝。
运行镜像摘要目前为 null；因此本包不代表 Linux GPU 镜像或跨机器动力学验收完成。

内容寻址缓存包含编译输入与工具摘要，复用相同 MJCF，以及未改变的文件内容块。
缓存损坏会失败。包中动力学与任务证据保持 `not_verified`，不会沿用旧环境的成功状态。
更改物理参数后的 MJCF 仍全量重编译；尚未实现引擎内局部对象重编译。

## 物理依赖图与失效

`build_dependency_graph(scene)` 生成实例、几何、物理参数、支撑、潜在接触、任务与产物节点。
`plan_scene_update(before, after)` 返回变化节点、依赖闭包、可复用节点及重验范围。
仅质量变化可以保留视觉节点；几何/位姿/支撑变化传播至相关产物。

本版没有证明有限运动包络覆盖所有新接触，因此任何环境变化均要求完整动力学/任务重验。
这是一条保守回退规则，不能用旧接触关系图证明局部验证充分，也不宣称已达到方案的四组
局部增量编译对照实验或专利新颖性要求。

## 受约束物理采样

`sample_instance_physics(scene, config, seed, registry=...)` 返回场景副本与采样记录。
支持动态叶子对象的 box/sphere/cylinder 代理，拒绝网格缩放、静态对象和未知物理字段。
场景原对象新增 `physics`，共享资产注册表保持不变；MuJoCo 导出器已实际应用这些实例覆盖。

```json
{
  "schema_version": "scene_factory.constrained_randomization.v1",
  "instances": {
    "part_1": {
      "scale": [0.95, 1.05], "density_scale": [0.8, 1.2],
      "friction": [0.2, 0.8], "scale_group": "paired_interface"
    }
  },
  "interfaces": []
}
```

每个参数使用独立的 seed 流。相同 `scale_group` 共用缩放抽样，组内范围不一致时拒绝。
密度比例为 d、长度比例为 s 时，质量按 s³d、惯量按 s⁵d 联动；高度调整保持初始底面。
可通过 `interfaces` 指定 peg/socket ID、原始半径和 `clearance_m` 区间，采样后重新计算间隙，
不能把独立缩放后的不兼容接口默认为可装配。

每个候选记录实际值、来源、配置摘要和拒绝原因。`sample_candidate_seeds` 分开返回
`parameter_valid/parameter_invalid`，物理有效、策略失败和任务成功统计在未执行时为 null。
不把不同 seed 自动当作无泄漏训练/测试划分。

本地证据位于 `outputs/delivery_20261028/environment_evidence`：100 和 1000 个固定候选
均通过当前参数约束；这是参数管线验收，没有运行这 1100 次候选的机器人控制。
复现脚本：`python tools/validate_environment_delivery.py <新的输出目录>`。

## 装配契约和原创夹具

`AssemblySpec` v1 采用米、秒、牛顿、弧度和 wxyz 四元数，描述 peg/socket 接口 SE(3)、
配合轴、目标深度、间隙、位置/角度公差、接触白名单、力限制、保持/释放稳定窗口及超时。
未知字段拒绝，不将未实现的装配顺序或柔性约束视为已支持。

`evaluate_assembly_trace(spec, samples)` 要求完整位姿、连续递增时间、几何间隙证据、
接触对象对及法向力、动作来源与释放状态。判据覆盖成功、卡住、偏斜、掉落、超力、
不允许的接触、超时、缺少证据和非法瞬移。目标保持与释放稳定必须是连续窗口；
初始化已装配、直接写位姿、缺少 robot action/socket contact 均不能算成功。
此函数是离线测量判据；合成测试轨迹通过不能证明真实机器人插接通过。

`tools/build_assembly_fixture.py` 在 pxr Python 中生成原创刚体 peg/socket/桌面场景、
`assembly_spec.json`、`layout.json` 和来源记录，未启动 GPU 仿真。Windows 运行时应使用
用户指定的 ASCII 输出目录，生成的夹具不依赖固定的开发机路径。
圆柱插销半径 10 mm，方孔半宽 10.75 mm，最小间隙 0.75 mm；使用四个碰撞墙和底板，
质量/惯量显式编写，源代码 MIT。其装配成功与求解器误差验收仍需独立运行证据。

## 20/50/100 物体房间基准

`room_benchmarks.generate_room_scene` 从现有客厅扩展背景物体，保留工作台区域，所有背景保留碰撞。
没有实例化 Franka，也没有把代理背景当作精细装配工作区验收。
`run_room_benchmarks` 每档执行 1000 个物理步，记录几何/动态体/接触规模、原始耗时和静置误差。
通过标准包含最后 100 步位置漂移 ≤5 mm、广义速度上界 ≤0.1、初始至最终位移 ≤10 mm 和有限状态。

最终本地证据：`outputs/delivery_20261028/room_evidence_final/room_benchmarks.json`。

| 物体 | 动态/静态 | 最大观测接触数 | step P50/P95 | 静置结果 |
| --- | --- | --- | --- | --- |
| 20 | 13/7 | 52 | 0.859/1.113 ms | 通过 |
| 50 | 37/13 | 148 | 2.429/3.055 ms | 通过 |
| 100 | 77/23 | 308 | 5.119/6.608 ms | 通过 |

三档最大初始至最终位移约 0.0492 mm；最后窗口漂移约 5.16e-15 m。
这是 Windows、MuJoCo、单进程、无渲染的开发机测量，未隔离其他后台负载。
没有测量 RAM/显存峰值，不承诺跨硬件相同耗时，不代表房间内机器人任务成功。
