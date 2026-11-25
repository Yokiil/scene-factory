# Isaac 环境包与 HTTP 验收

日期：2026-11-04。新增本地 USD 依赖冻结和 Isaac HTTP 验收入口。本文描述当前实现，
不将静态包校验、单元测试或失败试验写成物理任务通过。Linux GPU 仍未验收。

## 构建和锁定范围

使用安装了 OpenUSD 和 Isaac Sim 的专用 Python；构建本身不会启动 SimulationApp。

```sh
python -m scene_factory.cli sandbox build layout.json --backend isaac \
  --usd /srv/inputs/scene.usd --robot-asset-root /srv/inputs/isaac-assets \
  --output /srv/runtime/environment
python -m scene_factory.cli sandbox verify /srv/runtime/environment
```

`--robot-asset-root` 中须存在 `Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd`。
构建真正解析 scene 和 Franka 的 USD 层、payload、纹理与资产依赖，将层中资产路径改写为
包内相对路径，复制并记录每个文件的 SHA-256。任何无法解析且未声明为引擎内置的依赖导致失败。
模型不同 variant 的 USD 依赖也由 OpenUSD 的依赖遍历收集。

运行锁还记录 Python/Isaac 版本、安装的 Isaac/Omniverse distribution 版本、引擎 extension
配置摘要、SceneFactory Python 源码摘要和可选 `SCENE_FACTORY_RUNTIME_IMAGE_DIGEST`。
`OmniPBR.mdl` 依赖由已安装引擎提供；除入口文件之外，还锁定整个 `kit/mdl` 文件树，
覆盖 ClearCoat、NVIDIA 模块等进一步引用。构建没有把引擎安装目录复制进资产包。
镜像摘要未提供时保持 `null`，不会编造镜像或宣称容器依赖已冻结。

因此这个包闭合的是本地场景/机器人资产，运行程序依赖仍由匹配的 Isaac 安装或部署镜像提供。
`sandbox verify --content-only` 只检查包内字节与摘要；运行校验另外核对本机版本、源码和
USD 实际解析范围。校验成功不是动力学或装配成功。

默认锁定 `enable_rgbd=true`、`enable_general_observation=true`、`physics_dt=1/60` 和
`max_steps=10000`。可用 `--backend-options settings.json` 显式冻结这四个参数。
`cache` 目前用于 MuJoCo 编译，Isaac 构建不据此宣称具有引擎编译缓存。

## 私有资产与源码交付

构建器使用操作者提供的本地官方资产缓存，不为其生成 MIT 或其他再分发许可声明。
运行锁将包标记为 `private_local_runtime_package`，并明确 `redistribution_rights=not_asserted_by_builder`。
本机包放在仓库之外；通用源码 ZIP 只交付构建代码、原创零件、报告和复现入口，
不应包含本机官方 Franka 缓存、复制出来的机器人资产或引擎文件。
资产使用及对外分发范围由取得资产的主体依据相应条款确认。

## HTTP 构建与交互

必须用 Isaac Python 启动服务。Windows 的 USD/运行包路径使用 ASCII；源代码路径不要求相同。

```sh
python -m scene_factory.cli sandbox serve --root /srv/runtime/service \
  --scene-root /srv/inputs --max-sessions 1 --startup-timeout 180 --request-timeout 180
```

`POST /v1/builds` 的 Isaac 请求：

```json
{
  "idempotency_key": "build-isaac-1",
  "backend": "isaac",
  "scene_path": "layout.json",
  "usd_path": "scene.usd",
  "robot_asset_root": "isaac-assets",
  "backend_options": {
    "enable_rgbd": true,
    "enable_general_observation": true,
    "physics_dt": 0.016666666666666666,
    "max_steps": 10000
  }
}
```

查询 build 完成后，将返回的 `package_dir` 用作创建请求的 `environment_path`，同时指定
`backend=isaac`。构建产物位于服务根目录内，因此服务允许自己的已验证构建产物作为输入。
会话再次复制并校验包；不能通过 `backend_options`、`registry_path` 或 `usd_path` 覆盖锁定内容。
worker 也重新检查冻结包的运行兼容性。

随后使用普通 reset/step/observation/DELETE 接口。`joint_position` 动作是七个 arm 目标和
0..1 夹爪开度，实际运行关节限位会再次校验。RGB-D 返回 NPY 描述符，可由
`RuntimeClient.artifact(session_id, name, sha256=descriptor["sha256"])` 下载并校验字节摘要。
图像与物理状态使用同一 `state_version`、`step_index` 和 `sim_time`。

## 可选观测和失败行为

直接构造 `IsaacSimBackend` 时，`enable_general_observation` 默认关闭；运行包默认开启。
开启后动态物体的 SE(3) 和线/角速度来自实时 PhysX tensor handle，静态物体由静态 USD
变换与零速度表达。所有可遍历、启用的 CollisionAPI shape 纳入 contact filters，具有
RigidBodyAPI 的参与体作为 sensor。静态体之间没有求解器动力学接触，不作为 sensor。

返回 `contact_coverage` 中的 collider 路径、数量、sensor 数量、逐 sensor 缓冲区占用和测量范围。
`normal_force_N` 使用 `get_contact_force_data` 的逐接触点法向力绝对值之和，即使相向力
在向量合力中抵消，也不会漏掉。`normal_force_resultant_N` 另存 shape/filter 合力模作为诊断。
反向测量取较大值避免双计数。缓冲区已满、索引越界、数据缺失或无法覆盖合力时，测量失败；
这些条件不能获得装配验收。对象内部和同 owner 的机器人内部接触不在对象对输出中。
摩擦力向量和软体接触明确标为未支持。

开启通用观测时，原有指尖接触 sensor 同样显式设置 `prepare_contact_sensors=false` 和
`disable_stablization=false`，保留作者设置的休眠/稳定化参数；实际选项记录在
`finger_contact_sensor_settings`。未开启的历史路径保留原传感器默认行为，二者的物理验收
应分别记录，不能直接假定数值轨迹相同。

通用观测初始化或读取失败时，请求明确失败，faulted 会话不能继续步进；不会静默退回旧观测。
`observation_capabilities` 记录 requested、pending/ready/failed 与错误。
初始化异常先传播给调用者保存证据，之后显式 close 回收 Kit，避免 fast shutdown 吞掉错误回执。
直接调用者若明确需要历史行为，可设置 `close_on_reset_error=True`；服务默认保留错误回执路径。

每个 Isaac worker 仍为一个 episode；原 reset 幂等键可以重试，新 episode 创建新 session。
`snapshot_levels=[]` 保持不变，通用观测和环境包没有增加 Isaac 任意状态快照能力。

## 在线装配判定

将完整版本化 `AssemblySpec` 放入 `task.assembly_spec`，并提供由对应几何计算/构建记录得到的
`task.geometry_clearance_m`；`task.success.predicate` 必须为 `assembly`，`target_object` 必须是
该规格的 `peg_id`。零件 ID、规格字段和间隙类型在启动 Kit 之前校验。

该路径仅接受外部 `joint_position` 控制。后端在初始状态及每个实际物理步调用
`AssemblyRuntimeEvaluator.record`，传入实时 SE(3)、速度、全部刚体接触、动作和实际释放状态。
`contact_coverage` 采用 `scope=all_scene_contacts`、`continuous=true`、
`force_aggregation=sum_per_point_normal_magnitudes`，并列出对象 ID、采样 dt 和测量源。
时间连续性由判定器再次检查；漏掉一个物理步、缺乏完整覆盖或逐点力测量会让本 episode
保持 `not_verified`，后续看起来正确的最终姿态不能恢复丢失证据。

释放不接收客户端布尔声明：需先有闭合请求期间的实时手指/peg 接触，随后开度请求达到
0.98、实际手指到达打开范围，且完整接触观测没有 robot/peg 接触。之后仍需通过冻结规格的
深度、横向/角度误差、保持时间、释放稳定时间、速度和力阈值。成功奖励只发一次。
每个 step 的 `assembly_status`/`robot.task_oracle` 返回判定和失败原因；
`backend.assembly_trace()` 返回所记录的原始证据副本，供本地验收工具导出。
会话在第一个动作之前把初始证据绑定到外层 API 的同一 `episode_id`；幂等 reset 不会重采样。

`task.controller` 支持 `finger_stiffness`、`finger_damping` 和 `finger_max_force`；默认分别为
1000、200 和 2000，与旧控制器相同。参数必须有限且为正，运行初始化后会读回并核对实际值，
通过 `runtime_drive_gains.fingers` 返回。修改控制参数不会改写冻结的装配力/速度验收阈值。

`task.controller.solver_position_iterations` 和 `solver_velocity_iterations` 可选配置机器人
articulation 的求解迭代次数；默认 `null` 保留引擎值。位置次数为 1–255，速度次数为 0–255，
必须是整数。初始化后通过 SingleArticulation 的运行 API 设置并读回核对，诊断位于
`runtime_solver_settings`。此记录是 articulation 层参数，场景级最小/最大迭代设置仍可能
约束有效求解次数。提高次数的精度和性能影响需要独立实测，不能代替任务验收。

`tools/run_precision_assembly.py --online` 会启用完整观测并注入原始装配规格。
收到成功或失败的终止回执即停止继续发送动作，导出 `online_trace.jsonl`、完整回执和实际驱动值。
这是本地 `RuntimeSession` 路径；报告明确标记传输方式，不将其冒充 HTTP 装配验收。
初始参考轨迹的实时角速度存在超出冻结阈值的采样，不能仅凭已通过的位姿差分判据宣布在线通过。

没有 inline 规格的旧 `assembly` 场景仍明确返回 pose-only evaluator 不支持该判据；
未开启通用观测时不会取得在线装配成功。默认关闭通用观测的直接后端调用保持可用于独立
物理试验，但其独立验收结果不能自动变成 API 的在线任务成功。

`tests/test_isaac_assembly_runtime.py` 的合成测量数据只验证接入契约、缺失证据拒绝和奖励语义，
不是 GPU 插接验收。完整在线物理任务仍需真实控制/接触轨迹验证。

## 源锁与验收协调

构建/运行期间不要修改参与锁定的源码。运行版本不匹配会明确报告
`runtime_mismatch:source_sha256` 和 `runtime_source_changed:<path>`；重新构建后再创建新会话。
服务不修改源码、不关闭版本校验来迁就开发中的文件变化。

管理进程在构建和创建会话时完整解析 USD 依赖闭包。新 Isaac worker 在启动 Kit 前使用
`resolve_assets=False` 重新检查清单摘要、所有依赖文件大小/SHA-256、必需路径和当前
引擎/源码/扩展/MDL 哈希，只跳过再次加载 USD 解析器。这样由 SimulationApp 负责 USD DLL
初始化，避免独立 pxr 提前加载造成 ABI 冲突。该选项没有跳过文件或运行版本校验。
`usd_dependency_resolution_checked` 明确记录是否执行了该次解析。

第二轮真实 HTTP 验收在 reset 时暴露了上述预加载冲突，失败回执保留在
`outputs/completion_20261104/isaac_http02/report.json`；它没有产出有效图像或状态，不能计为通过。

本轮第一次 HTTP 验收在启动 GPU 前就因并行源码变化被正确拒绝，失败报告保留在
`outputs/completion_20261104/isaac_http01/report.json`，不能将其计为通过。
源码稳定后使用新目录运行以下工具；该工具显式开启通用观测：

```sh
python tools/validate_isaac_http.py layout.json scene.usd \
  --asset-root /srv/inputs/isaac-assets --runtime-root /srv/runtime/http-acceptance \
  --output outputs/completion_20261104/isaac_http_final
```

工具经真实 HTTP 完成异步资产构建、锁定包会话、reset、外部关节动作、幂等重试、
对齐 RGB-D、图像下载校验和资源回收，并保存完整回执及耗时。
通过该检查只证明本机服务链路，不替代装配成功、Linux GPU 部署或性能对照验收。

2026-11-04 的第三轮真实验收已经通过，完整回执和下载数组位于
`outputs/completion_20261104/isaac_http03/`。新包摘要为
`0d0e44020236c46c1c6f95ae7b593be31467bcf562b5e3c0e0e5d5eac61b12fc`。
构建耗时 3.763 s，新 worker 创建到初始观测 29.183 s，5 步关节动作响应 57.4 ms；
这是单次本机样本，不能用作 P50/P95 或 Linux 性能结论。
该场景测得 44 个碰撞形状、12 个刚体接触 sensor，逐点力 buffer 未饱和。
未覆盖写入 solver override 时读回的 articulation 次数为位置 32、速度 1，两个字段均已核对。
RGB 为 160×120×3 uint8，深度为 160×120 float32 米；下载大小、SHA-256、类型和形状验证通过。
预览显示一致的桌面与杯子轮廓，RGB 低分辨率且有采样噪声，机械臂位于相机裁剪范围之外。
worker 已回收，退出码为 0。`artifact_audit.json` 保存了数组审计和预览哈希。

## 显式的碰撞几何物理估算变体

`task.physics_overrides` 是可选的 `scene_factory.robot_physics_overrides.v1` 配置，schema 位于
`schemas/robot_physics_overrides.schema.json`。示例为
`outputs/completion_20261104/corrected_physics_config.json`，可由精细插接工具的
`--physics-overrides` 参数加载。旧场景未提供该字段时不启用任何覆盖。

本机实测发现，当前供给的 Franka 资产在 live PhysX 中使用的指尖惯量为约
`[4.204e-10, 3.903e-10, 1.335e-10] kg m²`，且其余 link 也保留极小的原写入值。
`diagnostics/live_franka_inertia_01.json` 保存了全部 link 的读回。几何尺度检查提示需重新
验证这些值；这不表示已获得实体 Franka 的实测质量分布。

覆盖在引用机器人之后、PhysX 初始化之前应用，仅写当前 stage 的匿名 session layer。
为选中机器人/任务部件的 collider 解除当前会话中的几何实例化，显式应用 PhysxCollisionAPI，
冻结有限 contact/rest offset 并读回。原始 USD 及共享资产缓存不保存、不修改。
每个 robot link 保留原资产的显式正质量，将惯量、质心、惯量主轴设为 USD Physics 的自动
碰撞体积计算模式。估算假设是碰撞几何上的均匀密度，**不是实体机器人或 CAD 的实测惯量**。
仅支持具有有效三维体积的凸碰撞网格；缺失质量、非有限/退化几何、不支持的碰撞类型明确拒绝。

初始化后读取实际 articulation tensor 的质量、完整惯量张量、质心和主轴，检查质量保持、
有限性、正定性、惯量三角关系和相对于几何尺度的保守合理性范围，失败则拒绝 reset。
`runtime_summary.physics_overrides` / step 的 `info.physics_overrides` 返回原值、几何哈希、
配置来源、匿名层摘要、接触及 solver USD 读回、实际张量读回和 `live_verified`。
接触 offset 读回是显式 USD API 值，不冒充自动 offset 的原生数值测量。

该配置还支持 robot solver、选中动态任务部件 solver、scene 最小/最大 solver 次数。
静态 socket 可以配置接触，但不能请求动态刚体 solver 次数。示例将 peg 与机器人设为
64/16，同时明确 scene 范围；控制器的显式 runtime solver 设置仍在后续设置并核对，需与
该配置保持一致。所有变化都有独立配置摘要；不改变任务的力、速度、误差和保持时间阈值。

Isaac 环境包运行锁覆盖全部 `scene_factory/*.py` 子目录源码，新增模块自动进入源锁，
配置进入冻结 layout。旧源码包不因此继续兼容，需重新构建。MuJoCo 环境包拒绝该 Isaac
专用配置。当前源码/合成张量测试只证明配置和接入，真正的变体惯量读回及插接能力仍须
使用新配置重新运行 GPU 验收；旧参考试验结果不能迁移到新物理变体。
