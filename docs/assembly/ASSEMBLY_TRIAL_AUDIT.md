# 已录制装配试验审计

`tools/audit_assembly_trials.py` 只读取录制文件，不启动 Isaac、GPU 或仿真，不修改原始 `report.json`。

```sh
python tools/audit_assembly_trials.py --root outputs/completion_20261104 --output outputs/completion_20261104/assembly_audit.json
python tools/audit_assembly_trials.py outputs/completion_20261104/assembly_online_01 --output outputs/single_assembly_audit.json
```

默认选择 `assembly_precision_*`、`room*_physics_*`、`assembly_online_*` 及独立 `square_assembly_*`、`square_room*_physics_*`、`square_room*_online_*` 目录；可重复使用 `--pattern` 指定其他目录模式。新增试跑完成后应重新运行，以把新证据加入聚合文件。

每条试验保留历史报告中的验收、进程状态与退出码，同时用该试验的冻结规格和当前判据源码重新计算位姿轨迹、逐帧实时线/角速度、末尾释放窗口速度、接触覆盖及实际松爪声明。在线轨迹优先于旧版离线轨迹，但两者都保留审计结果。当前已知失败返回 `failed`；未发现失败但缺少完整证据返回 `not_verified`；全部规定的保存证据通过才返回 `passed`。

报告记录原始轨迹、规格副本、控制器源码快照、原始报告与当前审计源码的 SHA-256。试验规格副本因 JSON 键排序而与原文件字节不同，可通过原夹具中 SHA 匹配的 `assembly_spec.json` 和完全相同的解析阈值绑定；阈值改变无法借此通过。夹具核验只说明输入文件哈希和声明间隙一致，不等于重新证明几何或物理有效。

旧版 peg 专用接触传感器缺少每步的两个任务零件完整覆盖声明，不能升级成完整在线验收。当前 Isaac 在线轨迹保存的是逐点法向力大小求和后的对象对测量及点数，并不包含全部逐点力向量。接触覆盖和实际松爪依赖保存的后端测量声明，审计程序不能凭文件独立证明传感器完整性。

命令退出 0 表示审计执行完成。它不表示试验通过；任务结论看每条 `current_audited_acceptance`。位姿差分通过而实时速度超限的试验会被明确标为失败。原始成功/失败报告始终保留，供复核判据与版本差异。

原始报告在引擎退出前写入，因此其中 `cleanup` 可能保留 `pending_process_shutdown`。外部监督者确认进程退出后，可另存 `supervisor_receipt.json`，声明 `scene_factory.assembly_trial_supervisor.v1`、试验目录名、退出码、`cleanup: process_exited`、观察来源，以及 `report.json` 和轨迹的 SHA-256。审计会绑定这些文件，但明确这是外部监督记录，审计器没有亲自监督进程；退出码仍不参与任务通过判定。原报告和轨迹均不改写。

另一台机器可以用可重复的 `--fixture-map "原目录或前缀=解压后目录"` 指定夹具位置；程序不会改写报告内路径或绕过哈希。映射后的 `scene.usda` 必须匹配原报告，`SOURCE.json` 的 USD 摘要必须匹配该文件；变体声明了 `frozen_spec_sha256` 时也必须匹配原报告。原始规格副本仍需通过字节哈希或同一冻结原文件绑定，几何或规格被篡改会拒绝完整通过。示例见 `examples/assembly/square_peg_socket/README.md`。

公开的 8 组原始夹具可使用 `--fixture-map-file examples/assembly/fixture_path_mappings.json` 一次读入。JSON 为 `scene_factory.assembly_fixture_paths.v1`；相对目标目录相对源码根目录解析。完整最小证据清单见 `ASSEMBLY_EVIDENCE_PACKAGING.md`。

房间内有动态背景时，另读 `background_trace.json` 重算任务末尾连续 1 s 的逐物理步测量。背景对象集合需与保存布局和初始状态一致，末帧需与任务对齐；固定阈值仍为漂移 1 mm、相对初始位移 5 mm、线速度 5 mm/s、角速度 0.05 rad/s。报告中的 `passed` 不代替复算。`local_assembly_audited_acceptance` 单独保存局部插接结论；背景超限使整体 `failed`，背景证据缺失/稀疏/未对齐使原本通过的整体结论变成 `not_verified`。这只核对末尾窗口，不外推到整段 episode。
