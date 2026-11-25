# MuJoCo 增量编译与四组对照交付

日期：2026-11-04。本轮将原先的依赖失效计划推进为可执行编译器，但范围是当前
MuJoCo 自由刚体代理场景。下面的推物接触任务不是 Franka 插接或维修任务验收。

## 已实现路径

`scene_factory/incremental_compiler.py` 提供四种不同执行路径：

| 模式 | 实际行为 |
| --- | --- |
| `full` | 每次重生成整个 MJCF，并调用 MuJoCo 原生编译器 |
| `file_cache` | 按整场景和资产依赖哈希缓存 MJCF；精确命中时跳过生成，仍进行原生编译 |
| `incremental` | 按对象及其共享资产依赖缓存片段；只生成失效对象，再执行安全参数更新或原生重编译 |
| `partition` | 增量路径加构建时固定背景碰撞表示分区；仅将可证明为长方体的固定凸网格换成同形状 box |

对象缓存键包含实例输入、资产记录、实际几何源文件和生成器源码哈希。一次请求对共享资产
只读取并哈希一次；不同对象仍有独立片段。内存对象缓存上限为 4,096 项，整场景缓存上限
为 128 项；可选磁盘缓存由部署目录的容量策略管理。损坏缓存拒绝使用。

增量模型更新只在 episode 边界进行，返回全新的初始运行数据。当前可复用已有模型的改变为：

- 直接挂在世界下的自由刚体完整初始位姿，包括显式 `orientation_wxyz`。
- 动态 primitive 明确给出的质量、主惯量与摩擦；没有明确惯量时不推测参数联动。

改变几何、固定物体姿态、拓扑、非受支持物理字段或求解器配置时，调用原生编译器。
更新前保存主数组，意外失败时恢复；更新后运行 `mj_setConst`、reset 和 forward。
代码保留每次更新类型、片段命中、失效原因、源码/引擎版本和耗时。

## 新接触和验证边界

`motion_envelope_query` 对旧、新两份布局重新查询候选物体对，不只遍历旧接触图。
球包络包含任意朝向的盒体，并与真实网格顶点半径取较大值；动态物体增加
`最大线速度 × 时间窗` 的移动距离。变化会沿可能交互的动态链传播；共享固定地板
不会被当成两个动态物体之间的运动传递关系。

速度上界是调用者条件，不能仅写入配置就认为求解器保证它。实验每个物理步检查实际线速度。
超出上界时该局部包络结论无效；生产调用方应全量回退或扩展包络。当前编译器不会复用旧的
动力学/任务成功证据，所有变体均要求重新执行；本轮实验四组每个变体都做完整 200 步运行。
尚未提供省略背景运行的局部动力学验收证明。

分区也在运行前固定。若待简化背景进入任一动态物体包络，构建失败；只允许固定背景中
恰为八个角点的凸长方体网格替换为 box。两种表示的接触算法仍可能不同，因此实验同时
比较状态轨迹、每个物理步的接触点与法向力；不承诺任意网格的几何简化或在线精度切换。

## CLI 和 Python 使用

```sh
python -m scene_factory.cli sandbox incremental-compile before.json after.json --output outputs/variants --cache outputs/object-cache
python -m scene_factory.cli sandbox incremental-compile before.json after.json --mode partition --background-id cabinet_1 --horizon-s 0.4 --max-linear-speed-mps 2 --output outputs/partitioned
```

目录保存每个变体的 `layout.json`、独立可编译 `scene.xml` 和 `compilation.json`。
命令只声明编译成功，`physics_acceptance` 与 `task_acceptance` 仍为 `not_evaluated`。

```python
from scene_factory.incremental_compiler import IncrementalMujocoCompiler

compiler = IncrementalMujocoCompiler(registry, mode="incremental")
first = compiler.compile(before)
second = compiler.compile(after)  # 同拓扑受支持参数可原地更新 model；data 是新 episode。
print(second.report["kind"], second.report["fragment_hits"])
```

该编译器是独立显式接口；当前默认 MuJoCo 后端仍保留原有兼容 reset 路径。CLI 产物不能自动
当成已完成动力学验收的环境包，旧版本环境锁也不会因有新编译器而被升级。

## 两组四臂实测

当前冻结输入与逐步力复核证据：`outputs/completion_20261104/incremental_locked_v2/`。两种变体分布各包含 30 个变体；
每个变体有 60 个固定网格背景、一个受力自由刚体和固定挡块。输入包含质量/惯量、摩擦、
初始位姿、尺寸、固定挡块位置和拓扑变化。所有策略使用相同输入、MuJoCo 3.3.7、
0.002 秒时间步、implicitfast、12 N 外力、200 步和无渲染设置。

每条完整状态轨迹以无 pickle 的 `.npy` 保存；每个物理步的所有接触点保存几何对象对、
点法向力大小、接触坐标系中的六维力/力矩、位置和距离，逐条附 SHA-256。
`verify_incremental_benchmark.py` 独立重读这些文件，比较状态、接触点、逐点法向力、
同一几何对的法向力大小之和、六维力/力矩分量、最终位置和速度条件；并非只比峰值。

`inputs/` 保存完整原始注册表字节与 SHA-256、实际使用的原始 JSONL 行、规范化资产记录、
所用几何与来源清单、按缓存键顺序冻结的编译器源码以及实验工具源码。四组实际从冻结的
选定注册表及其几何目录加载。实验结束再次检查全部文件哈希、源码闭包与所消费资产记录；
独立验证器再次复核。完整注册表 SHA-256 为
`57158c736be5c3c58b32f40cc0b86a3cc8c4da979f494f590da606823d90f660`，编译源码闭包为
`bcadf2bc292b65d038f104e3d4057f767da024c85477fa05ee60644e545441b6`。

| 输入分布 | 全量 A P50 | 整文件缓存 B P50 | 增量 C P50 | 固定分区 D P50 |
| --- | ---: | ---: | ---: | ---: |
| 重复变体，B 命中 21/30 | 50.330 ms | 15.831 ms | 15.319 ms | 17.207 ms |
| 30 个不同参数变体，B 无命中 | 37.522 ms | 36.799 ms | 15.051 ms | 12.991 ms |

这些是生成、依赖处理和引擎更新合计的本机测量。C、D 各有 15 次参数更新和 15 次原生回退；
A、B 均有 30 次原生编译。四组各 30/30 通过这项代理任务，状态序列相对 A 的最大差异为 0，
逐步接触对完全相同，速度上界均满足。两份独立验算报告分别验证 120 条轨迹、24,000 个物理步，
比较 97,704 / 99,732 个接触点。逐点法向力、对象对法向力之和、力/力矩分量和接触位置最大差异均为 0。

不同参数变体中 C 的构建 P50 比 A 低约 59.9%；包括完整动力学重验后，P50 从 54.288 ms
降至 31.759 ms。重复输入时 B 与 C 接近；此前历史试跑中 B 更快，不能声称依赖增量总优于普通缓存。
D 在重复输入中比 C 慢、不同参数集合中更快，本实验未证明稳定额外收益。

`incremental_complete/` 等早期目录保留历史结果；其接触文件只有对象对和峰值，不能用于声明
逐点力序列已经比较。`incremental_v2_smoke/` 使用 100 步短时运行，未达到最终目标而失败；
`incremental_v2_smoke200/` 以 200 步通过，仅是正式冻结实验前的工具冒烟。

同机同时可能存在其他任务，实验未隔离 OS 负载，未测峰值 RAM。原始逐次时间、P95、
模型规模、源码哈希、完整轨迹及失败的早期调试实验均保留，不外推到 Isaac、机器人接触
精度或服务器吞吐。

```sh
python tools/benchmark_incremental_compiler.py --output outputs/recheck_repeated --variants 30 --backgrounds 60 --steps 200 --pattern repeated
python tools/benchmark_incremental_compiler.py --output outputs/recheck_unique --variants 30 --backgrounds 60 --steps 200 --pattern unique
python tools/verify_incremental_benchmark.py outputs/recheck_unique --output outputs/recheck_unique/verification.json
```

## 与候选技术机制的关系

已有对象依赖、局部生成、安全模型参数更新、新接触包络查询和四组对照的可测切片。
仍需研究复杂关节/机器人环境的局部重编译、具有充分条件的局部动力学验证，以及任意背景
网格精度分配；本轮结果不构成专利新颖性结论。完整性和任务证据继续保守失效，不能为了
加速跳过未证明可省略的物理检查。
