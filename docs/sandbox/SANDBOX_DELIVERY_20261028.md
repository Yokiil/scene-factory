# SceneFactory 物理沙盒交付说明

日期：2026-10-28。交付当前可运行源码、CLI/HTTP/Python 接口、部署文件、测试和原始实验记录。
**原方案尚未全部验收完成。** Linux GPU 服务器目前不可用，按用户要求交付部署包并标明未验收。
真实插接已开展物理试验，但未通过冻结的完整装配判据；不能作为“精细装配平台完成”的证明。

## 从这里开始

交付 ZIP 解压后进入 `scene-factory-sandbox/`。Python 3.12；核心编译只用标准库，
MuJoCo 运行需安装可选依赖。参考开发环境为 Windows、Python 3.12.7、MuJoCo 3.3.7。
Linux 环境依赖固定在 `deploy/sandbox/requirements-mujoco.txt`；Isaac 必须使用其专用 Python。

```sh
python -m pip install -e ".[dev,mujoco]"
python -m scene_factory.cli sandbox capabilities
python -m scene_factory.cli build --recipe living_room_returned_home --seed 19 --output outputs/example_scene
python -m scene_factory.cli sandbox build outputs/example_scene/layout.json --output outputs/example_environment --cache outputs/compiler_cache
python -m scene_factory.cli sandbox verify outputs/example_environment
python -m scene_factory.cli sandbox smoke outputs/example_scene/layout.json --backend mujoco --steps 5 --output outputs/local_smoke.json
python -m scene_factory.cli sandbox serve --root outputs/runtime-service --scene-root outputs
```

最后一条命令启动常驻服务；客户端与请求示例见 [HTTP 服务](SANDBOX_SERVICE.md)。
`sandbox create/reset/step/observe/snapshot/restore/close` 调用同一核心；输入与动作校验失败明确返回错误。
旧的 `build`、`dataset` 和 `.scene.zip` v1 保持可用，新环境清单使用独立 schema。

## 对照原方案的交付状态

| 工作项 | 交付和实际证据 | 未完成或未验收部分 |
| --- | --- | --- |
| 会话契约与重置 | 版本化 schema；episode/sequence/幂等键；动态叶子实例 XY/yaw reset；seed、实际采样与摘要 | Isaac 新 episode 仍须新 worker；通用任意 SE(3) 布局迁移未完成 |
| CLI/API/Python SDK | HTTP、持久生命周期、异步构建、独立常驻进程、超时/租约回收、输入冻结、图像下载 | 多租户身份/归属、管理器崩溃后的自动接管、可复用空闲 worker 池未实现 |
| 外部动作 | Isaac Franka 7 关节位置目标、夹爪运行时限位语义；实际驱动 smoke；同键重试不重复步进 | 速度/力矩模式；现有 scripted 控制器全面改为标准接口客户端未完成 |
| 观测 | MuJoCo 全位姿、速度、接触、时间；Isaac 关节/末端/夹爪接触、按需 RGB-D 与状态对齐 | Isaac 全物体速度/任意接触归一化、分割、多相机/外部神经渲染适配未实现 |
| 环境包 | MuJoCo 完整代理几何、依赖哈希、源码/引擎/求解器设置锁定、验证与 ZIP；缓存内容校验 | Isaac USD/机器人/插件运行闭包打包；镜像真实摘要待部署生成；非所有后端的最终编译惯量都已输出 |
| 两级恢复 | dry-run 初始化重建；MuJoCo 初始化与运行状态恢复，重复恢复幂等 | Isaac 状态快照；跨硬件/引擎一致性；任意复杂接触恢复保证未提供 |
| 实例随机化 | 独立 RNG、质量/惯量/尺度相关采样、实例覆盖、100/1000 候选完整记录 | 四层随机化尚未全接入 episode；卡滞故障、传感器噪声、完整资产族训练划分及参考策略分层成功率未验收 |
| 20/50/100 房间 | 代理房间规模、动态体、接触数、1000 步静置与耗时报告 | 房间内机器人工作台装配、任务工作空间、精度分区与导航未验收 |
| 两件式装配 | AssemblySpec、完整接口姿态、接触/速度/保持/释放判据、正负合成测试、原创 USD 和真实 Isaac 尝试 | 真实成功案例与负例全套验收未通过；合成判据测试不代表物理成功 |
| 性能 | 分段本地基准；MuJoCo 模型复用；100 reset 对照、1000 reset 检查、恢复轨迹误差；HTTP 服务测量工具 | Isaac 温重置/向量化/每 GPU 并发和显存硬配额；完整下载到写盘全链路计时未完成 |
| 物理依赖图 | 几何/参数/支撑/任务依赖表示、影响闭包、失效理由、保守全量证据失效 | 真正局部引擎重编译、保守运动包络查询后局部验证、四组性能/精度对照未完成 |
| 部署 | CPU/GPU Dockerfile、Compose、doctor/probe、健康检查、资源配置、headless smoke | Linux 镜像构建、NVIDIA/驱动兼容、服务器任务均未验收；本机 Docker daemon 未运行 |
| 技术交底 | 技术机制草稿、边界和实验协议 | 现有技术检索、新颖性、专利结论均未完成 |

这里的“常驻”指一个会话多次调用使用同一个 worker；不等于会话之间自动复用引擎。
资源请求的 CPU/RAM 值是准入预算，硬限制由容器或部署平台提供。
MuJoCo 是当前支持的碰撞代理任务后端，其吞吐不等价于 Isaac 机器人接触精度。

## 本次物理与性能证据

证据目录为 `outputs/delivery_20261028/`；所有结果都限于报告中记录的硬件、引擎和输入。
`tests_report.json` 记录最终整套测试结果：**788 passed、2 skipped、1,408 subtests passed**，
退出码为 0。跳过项不计为通过；初次两项测试隔离问题及修复前日志另行保留。发布检查全部通过。

| 证据 | 实测范围 |
| --- | --- |
| `isaac_smoke/report.json`、`isaac_smoke_final/report.json` 和 `artifacts/*.npy` | Isaac 6.0.1.0，5 步外部关节控制；重试未重复执行；160×120 RGB/深度与第 5 步状态对齐；最终复测包含同状态重复采集 |
| `http_service/service_smoke.json` | 实际 HTTP 和 MuJoCo 子进程；创建/请求重试、常驻 worker、恢复、关闭等 8 项检查通过 |
| `mujoco_reset_restore_validation_final.json` | 最终 100 次 reload/reset 对照中位数约 2.143/0.663 ms；1000 次 reset 零状态漂移；100 步恢复续跑 qpos/qvel 差异为零 |
| `http_service_benchmark_final.json` | 30 次新 worker 到首观测 P50/P95 708.55/772.57 ms；100 次同 worker reset 20.54/32.55 ms；100 次单物理步 HTTP 14.85/33.42 ms；零失败 |
| `room_evidence_final/room_benchmarks.json` | 20/50/100 个物体、13/37/77 个动态体、52/148/308 个接触；每档 1000 步静置通过；步进中位数约 0.859/2.429/5.119 ms，无渲染/机器人 |
| `environment_evidence_final/` | 与最终代码匹配的包锁、完整包、缓存命中、100/1000 参数候选统计；任务/动力学结果明确未评估 |
| `assembly_trial_*/report.json`、`trace.jsonl` | 真正关节驱动、PhysX 实时位姿和接触测量；包括失败过程，不以最终插入位置代替完整判据 |
| `asset_dependency_closure.json` | OpenUSD 实际解析本仓库 42 个根层与 108 个依赖文件；不含外部官方 Franka 资源 |
| `deployment_*.json` | Compose 解析、本机预检、本机 MuJoCo HTTP smoke；不能外推为 Linux GPU 验收 |

最终 reset 对照的中位耗时下降约 69.1%，来自本机小场景；不承诺所有环境同样加速。
1000 次重置观察中 private bytes 未增加、working set 增加 102,400 字节，不据此宣称永久无泄漏。
HTTP 数据包含本机回环、JSON、状态持久化和进程通信；管理器常驻且 OS 资产缓存保留，
测量的是新仿真 worker 的创建，不是无缓存主机冷启动。空闲 worker 温创建未实现，不报告该指标。
精度和成功率必须与每条后端的测量一起看，不能将不同保真度吞吐直接比较。

装配试验的几何间隙、公差和阈值来自 `examples/assembly/peg_socket/assembly_spec.json`。
试验不写入运行中零件位姿，不吸附；初始化后只发关节驱动目标。
第二轮已完成抓取、移动、插入、松爪与撤离，但速度/姿态等证据未满足冻结判据。
最终横向误差约 0.819 mm（阈值 0.5 mm）、姿态误差约 0.345 rad（阈值 0.02 rad），
仪器记录的接触合力峰值约 857 N（阈值 30 N）。第三轮未保持装配并掉落。
三轮均按失败交付；当前试验脚本已加入测得超力后的停止逻辑。后续应先解决夹持滑移、
精确跟踪和完整接触覆盖，再扩展场景。历史试验的控制脚本副本保存在各自证据内，
不能将更新后的控制策略效果倒算到旧试验。

## 验证与复现入口

```sh
python -m pytest -q -o addopts='' --tb=short
python tools/check_release.py
python tools/smoke_runtime_service.py --output outputs/recheck_http
python tools/validate_mujoco_state.py --help
python tools/validate_environment_delivery.py --help
python tools/benchmark_runtime_service.py --help
python tools/run_assembly_trial.py --help
```

Isaac 复现需要安装对应引擎和合法可取得的 Franka 资产；参考机器上的绝对路径仅是
原始实验记录，使用者通过命令参数替换为自己的路径。原创 peg/socket USD 已随源码提供。
Linux 部署步骤、镜像摘要和许可配置见 [部署说明](SANDBOX_DEPLOYMENT.md)。

解压后可用 `tools/package_sandbox_delivery.py verify 交付包.zip` 检查每个归档文件的 SHA-256，
归档内 `DELIVERY_MANIFEST.json` 列出源码、依赖和证据；旁边 `.sha256` 校验整个 ZIP。
此 SHA 是完整性校验，不是签名或信任凭证。打包排除密钥、私有 LLM 配置、虚拟环境、
本机官方资产缓存、数据库和临时日志。

## 相关说明

- [环境包、随机化、依赖图与装配契约](../data/ENVIRONMENT_COMPILER.md)
- [MuJoCo 模型复用与恢复限制](MUJOCO_RUNTIME.md)
- [HTTP 服务与 Python 客户端](SANDBOX_SERVICE.md)
- [服务器部署包](SANDBOX_DEPLOYMENT.md)
- [增量构建技术交底草稿](../patent/INCREMENTAL_BUILD_DISCLOSURE_DRAFT.md)
- [原始设计方案](PHYSICS_SANDBOX_PLAN_20261028.md)

`SANDBOX_PHASE1.md` 和 `SANDBOX_BASELINE_20261028.md` 保留第一阶段的历史记录。
新增能力以本交付说明、当前 capability API 和对应新证据为准。
