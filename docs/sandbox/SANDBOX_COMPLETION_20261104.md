# SceneFactory 物理 AI 沙盒补齐交付

日期：2026-11-04。对应老师提出的真实接触、精细构建、CLI/API、快速创建环境、返回图像和物理状态，以及与 JEV 决策模型结合的方向。本轮在既有交付上继续实现并运行验收；2026-10-28 的方案和交付记录作为历史保留。

核心定位已经落实为环境编译器与有状态运行服务。源码、部署文件、可复现夹具、接口、测试与原始成功/失败记录共同交付。**不把源码齐备、合成测试通过或单条轨迹成功等同于整个研究计划全部完成。** 用户尚无 Linux GPU 服务器，因此部署包按约定交付，目标服务器验收仍未进行。JEV 接口已实现，真实模型调用与训练未执行。

老师要求的沙盒设计已对应为：构建产物与交互会话分开、冻结输入与依赖、按需创建独立 worker、
用动作推进状态、按需返回传感器产物，并保存任务判定和失败证据。这些具体机制是本项目对
Harbor/DSec 设计方向的落实；运行协议与依赖图实验见下文。自动化、CLI、容器本身不作为创新结论，
微波炉维修、移动操作和复杂故障建模也没有因基础接口交付而自动完成。

本轮最终源码回归：**962 passed、6 skipped、1489 subtests passed**；Ruff 与发布资源/许可证检查均通过，
检查前后源码摘要一致。原先的短租约慢启动失败已经修复，旧失败报告保留。
详见 `outputs/completion_20261104/release_checks_final_v4/report.json`。这些源码检查不替代物理任务验收。

`release_checks_final_v2` 因 C 盘临时空间耗尽失败，并暴露录制器初始化异常时未关闭句柄的问题；
该缺陷已修复，并覆盖磁盘耗尽、序列化失败和中断三种异常。`v3` 改用中文临时目录后触发
既有 Isaac/OpenUSD 的 ASCII 路径契约，仍记录为失败。最终 `v4` 使用纯英文
ASCII 临时目录 `<ascii-temp>\20261104_v4` 作为 TEMP/TMP 完整重跑通过，旧日志及分类审计全部保留。

## 入口与复现

解压交付 ZIP，进入 `scene-factory-sandbox/`。参考环境为 Python 3.12.7、MuJoCo 3.3.7、Windows；Isaac 6.0.1.0 使用单独解释器。

```sh
python -m pip install -e ".[dev,mujoco]"
python -m scene_factory.cli sandbox capabilities
python -m scene_factory.cli sandbox schema
python -m scene_factory.cli build --recipe living_room_returned_home --seed 19 --output outputs/example_scene
python -m scene_factory.cli sandbox build outputs/example_scene/layout.json --output outputs/example_environment
python -m scene_factory.cli sandbox verify outputs/example_environment
python -m scene_factory.cli sandbox serve --root outputs/runtime --scene-root outputs --mujoco-worker-pool
```

长时间构建通过 `/v1/builds`，交互通过 `/v1/sessions`；CLI 与 Python 客户端调用同一服务。完整请求示例见 [服务文档](SANDBOX_SERVICE.md)。服务默认本机单用户，CPU/RAM 采用准入预算；容器已配置部署时的进程组限制，目标平台上的实际强制效果尚未验收。

## 按需求验收

| 需求 | 已实现 | 验收范围或边界 |
| --- | --- | --- |
| CLI/API 快速环境 | 异步构建、内容哈希环境包、独立会话、持久生命周期、租约、超时和故障回收、Python SDK | MuJoCo 的真实 HTTP 集成与跨会话兼容 worker 池已有实测；Isaac HTTP 和 Linux GPU 状态分别见本轮报告与部署说明 |
| 外部 agent 控制 | 有界关节位置目标、实际运行时限位与夹爪语义、episode/sequence/幂等键 | 首种位置动作已实现；不能把 MuJoCo 外力代理任务算作机械臂抓取 |
| 图像与必要状态 | 可对齐的 RGB-D 引用、标定、时间、状态版本；MuJoCo/Isaac 全位姿和速度；法向力与接触覆盖说明 | Isaac 对逐点法向力大小求和后按对象对返回；没有保存完整逐点向量。按后端能力表拒绝不支持模态 |
| 版本化物理环境 | SE(3) 四元数、旧 yaw 迁移、实例质量/惯量/摩擦；输入/引擎/设置/源码锁 | MuJoCo 完整代理包；Isaac 私有资源闭包可在有授权资产的机器构建，通用源码 ZIP 不携带本机官方资产缓存 |
| 两件式精细装配 | 原创 peg/socket、原始固定间隙、公差、机器人驱动、接触/力/保持/释放判据与在线任务状态 | 原始失败、位姿差分通过记录、实时速度审计与新的在线验收分别保存，不互相冒充 |
| 房间工作台 | 20/50/100 物体夹具，原工作区与装配规格保持一致，保守工作空间分隔 | 背景静置、局部装配和规模性能分别报告；没有移动底盘/导航能力声明 |
| 四层随机化 | 布局、关联物理参数、实际初始错位、独立观测噪声；采样值与 RNG 记录 | episode 管线在 MuJoCo 实测；不支持的拓扑/资产替换、Isaac 动态随机化和真实电气故障明确拒绝 |
| 增量编译机制 | 物理依赖图、保守新接触查询、片段缓存、同拓扑实例参数/初始位姿更新、必要时全量回退 | 四组真实对照，仍完整重跑动力学/任务验证；没有宣称只看旧接触图即可复用验收结论 |
| JEV 结合 | 官方 System One 传输协议、Choice/Score/Noul 校验、有限技能选择、风险与拒答、真实沙盒 rollout、训练样本出口 | 未提供 TypeSafe key；真实模型推理、微调与泛化评估均未验收；接口替身明确标名 |
| 服务器部署 | CPU/GPU Dockerfile、Compose、依赖固定、doctor、headless HTTP smoke、资源与资产挂载 | 无 Linux GPU 服务器；本机容器运行也未验收。不得把 Windows GPU 验收外推为 Linux 成功 |

## 已完成的可测机制

MuJoCo 可选池复用同一进程和编译模型，新建独立会话、episode、请求历史与 MjData。兼容键覆盖场景、完整 MJCF、注册表、运行源码、引擎及二进制、设置与资源请求；错误、污染或不兼容时销毁/新建。Isaac 不进入此池。

`worker_pool_release_final/report.json` 中，30 次新 worker 创建到首观测 P50/P95 为 0.761549/0.869964 s；30 次兼容池复用为 0.138086/0.163286 s，中位数降低 81.87%。池复用测到同一 PID、总编译次数 1。1,000 次外力扰动/reset 的状态分量漂移为 0，HTTP reset P50/P95 为 12.489/33.567 ms。私有内存首尾增加 4,096 字节，工作集增加 126,976 字节。仅限单动态代理物体、无渲染、本机回环；OS 背景未隔离，不能外推为 Isaac 吞吐或永久无泄漏。

增量实验使用冻结注册表原始字节、实际使用行、资产和编译源码。`incremental_locked_v2/` 含重复/不同参数两种分布，各 30 变体×4 组×200 步。不同参数组编译 P50 为全量 37.522 ms、增量 15.051 ms；含完整重验合计 54.288→31.759 ms。状态、逐点法向力、对象对力和及力矩分量与全量差异为 0。固定精度分区未显示稳定额外收益；仅限本次 MuJoCo 刚体代理任务。

上述数字对应报告内冻结版本。后续代码变化需要重新构建包、核对源码锁，并按影响范围重新验收，不复用旧结论。

当前源码的 Isaac 真实 HTTP 验收位于 `isaac_http05/`：9 项检查通过，包含异步构建、完整资产闭包
校验、创建/重置/步进重试、外部关节驱动、完整位姿/速度与接触覆盖、RGB-D 文件下载与状态
对齐、worker 回收。单次构建 5.110 s，新 worker 到首个观测 26.163 s，5 步响应约 56.8 ms；
本次与源码回归同时运行，样本量为 1，不是冷启动 P50/P95。RGB-D 为 160×120，预览保留真实噪声，相机范围内可见
桌面与杯子，机器人在画面外。此验收使用抓杯参考场景的少量外部动作，未执行完整抓杯任务。
`isaac_http03/04/` 保留为此前源码的成功记录；这些单样本不作为启动加速实验。

Isaac worker 在启动 Kit 前使用不加载 USD 解析器的哈希/版本校验；管理进程完成完整 USD
依赖解析。该拆分修复了独立 OpenUSD 与 Kit 的 DLL 初始化冲突，并保留篡改拒绝与源码锁检查。
`isaac_http01/02` 的失败记录保留，不计入通过结果。

## 精细接触的证据口径

原始规格位于 `examples/assembly/peg_socket/assembly_spec.json`：几何间隙 0.75 mm、位置公差 0.5 mm、角度公差 0.02 rad、峰值法向力 30 N、保持与释放稳定各 0.5 s。初始化后只发关节驱动目标，不吸附、不瞬移零件。

`assembly_precision_07/report.json` 的位姿差分判据通过：最终横向误差 0.338 mm、角度误差 0.0034 rad、峰值法向力 18.4 N、末尾连续稳定约 4.77 s。后续审查发现其 PhysX 瞬时角速度超过冻结门槛，因此不能追认为完整在线验收。该结果、原始轨迹和审查报告均保留。新的工具同时审计实时速度；在线 evaluator 还要求逐物理步完整接触覆盖和实际松爪证据，缺失时保持 `not_verified`。

`tools/run_precision_assembly.py --online` 使用正式会话动作与在线判据，收到终止立即保存证据；`backend.assembly_trace()` 可导出原始在线输入。完整结论以 `outputs/completion_20261104/assembly_audit.json` 为准：历史位姿通过不替代实时速度、完整接触覆盖和实际释放证据。房间和负例的最终状态以各自报告为准，不以程序退出码代替任务验收。

最终逐条审计共 30 条：**0 通过、27 失败、3 未验证**。未验证为 `assembly_precision_01/02`
与初始化失败、没有动作样本的 `square_assembly_online_03`。32 个公开夹具文件在独立副本中重新
映射复核，原始证据摘要与各条结论均一致；可在无本机私有几何目录的情况下复查已有轨迹。
这是保存证据的可移植复核，没有重新执行物理试验。7 份新增进程 receipt 只追加兼容字段，
历史 15 条缺少可靠退出记录的状态继续保留。

本轮进一步读回了本地 Franka 资产的全部 link 质量与惯量。两根手指实际使用约
`[4.204e-10, 3.903e-10, 1.335e-10] kg·m²`，同质量、同包围尺寸的均匀体估算约大
一万倍。该诊断绑定本机资产组合与 PhysX 版本，不能泛化成所有 Franka 资产都存在相同问题；
几何估算也不等于真实机器人实测惯量。原始读回位于
`diagnostics/live_franka_inertia_01.json`，它只验收参数读取，没有发送机器人控制动作。

已新增默认关闭的 `task.physics_overrides`：保留各 link 质量，在匿名 stage layer 中按碰撞体
均匀密度重算惯量、质心与主轴，并显式设置接触距离与求解配置。初始化后核对真实张量和生效设置，
不写共享缓存。它是标为 `geometry_estimate` 的独立参数变体，不是实测机器人参数。
配置、原值和真实读回均随证据交付；[参数估算说明](../isaac/ISAAC_PHYSICS_ESTIMATES.md) 记录用法与边界。

修正参数后的 7 次记录见 `corrected_contact_trials.json`：一次初始化检查失败、六次真实动作试验
仍触发原始 PhysX 角速度门槛，**没有完整装配通过**。其中同一控制配置对 16/4/1 次速度迭代
各测一次，均失败；不据此给出统计排名。另有夹持增强、轻夹微抬升和原圆柱复测，全部保留。
新参数读回已解决惯量数量级问题，但没有解决控制与接触稳定性的全部问题。

Isaac 房间背景静置已分别验收：每档单进程、5 s 仿真、`dt=1/120 s`、600 次正式会话步进，
固定机器人默认关节位置、不渲染、不执行装配。采用显式估算参数与 64/1 求解设置，原夹具和
规格字节保持不变。三档最后连续 1 s 的背景漂移与原始速度均为 0，初末最大位移小于 1 µm。

| 物体数 | 动态物体数 | 背景静置 | step P50/P95 |
| --- | --- | --- | --- |
| 20 | 7 | 通过 | 16.137 / 18.798 ms |
| 50 | 17 | 通过 | 24.423 / 30.216 ms |
| 100 | 33 | 通过 | 34.876 / 39.912 ms |

时延包含完整观测、接触读取和幂等历史，排除启动、reset、轨迹写盘与渲染；每档是一次运行中的
600 个调用样本，后台负载未隔离，不能与纯物理内核吞吐等同。原始记录在
`isaac_background_room20/50/100/`，范围明确为 `background_settling_only_not_robot_task`。
三档完整机器人插接仍未通过。

## JEV 适配与训练边界

用户提供的 JEV 榜单适合借鉴有界结构化决策：由当前物理测量、候选技能和任务约束选择抓取、重对齐、撤回或中止，再交给下层控制器执行。[JEV 核对与案例](../assembly/JEV_ALIGNMENT_20261104.md) 保存官方协议、公开模型卡固定 revision 和案例借鉴边界。

本轮 12 个真实 HTTP/MuJoCo 随机 episode 导出 512 行训练样本，技能标签为实际执行行为，未证明最优。决策提供者明确为接口测试替身。数据共享一个资产族/任务结构分量，没有独立测试集，不能据此报告模型泛化。导出与校验说明见 [训练数据出口](../data/DECISION_DATASET.md)。

## 仍未完成的实现与验收

精细插接尚需解决夹持与接触稳定性，达到冻结的真实速度、力、精度和释放条件后，才能宣布通过。
20/50/100 房间内完整插接和完整实测负例矩阵也尚未通过；背景静置、接口测试及历史位姿结果
均不能替代这一闭环。这些属于尚未完成的软件与物理验证工作，不能归因于缺少 Linux 服务器。

Linux NVIDIA 部署和真实 OCI 镜像摘要需要目标服务器；JEV 推理需要可用端点/凭证；模型训练需要独立资产族/任务数据及计算预算。
Isaac 温重置/任意接触 checkpoint、GPU 向量化与显存硬配额、移动维修任务、多用户会话归属仍不属于已实现并验收的能力。
专利稿提供可测机制与实验，不提供新颖性结论。

## 文件索引

- [HTTP 服务](SANDBOX_SERVICE.md)、[MuJoCo 池](MUJOCO_WORKER_POOL.md)、[部署](SANDBOX_DEPLOYMENT.md)
- [Isaac 运行包与观测](../isaac/ISAAC_BUILD.md)、[在线装配 API](../assembly/ASSEMBLY_RUNTIME_API.md)
- [显式物理参数估算](../isaac/ISAAC_PHYSICS_ESTIMATES.md)、[Isaac 背景静置验证](../isaac/ISAAC_ROOM_BACKGROUND_VALIDATION.md)
- [SE(3) 与实例参数](../assembly/SE3_POSES_20261104.md)、[房间夹具](../assembly/ASSEMBLY_ROOM_FIXTURES_20261104.md)
- [增量编译与对照](INCREMENTAL_COMPILER_20261104.md)、[技术交底草稿](../patent/INCREMENTAL_BUILD_DISCLOSURE_DRAFT.md)
- [2026-10-28 历史交付](SANDBOX_DELIVERY_20261028.md)、[原方案](PHYSICS_SANDBOX_PLAN_20261028.md)

通用交付 ZIP 内的 `DELIVERY_MANIFEST.json` 列出全部源码和选中证据的 SHA-256；旁边 `.sha256` 校验整个 ZIP。验证：`python tools/package_sandbox_delivery.py verify <archive.zip>`。没有把官方 Isaac 私有资源、凭据、虚拟环境或数据库打入源码 ZIP。
