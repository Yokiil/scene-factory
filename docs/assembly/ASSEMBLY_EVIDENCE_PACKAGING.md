# 装配证据的最小交付集合

审计从交付包解压后的项目根目录运行，不要求取得官方机器人资产或重新启动 Isaac：

```sh
python tools/audit_assembly_trials.py --root outputs/completion_20261104 --fixture-map-file examples/assembly/fixture_path_mappings.json --output outputs/completion_20261104/assembly_audit.json
```

路径映射文件包含 8 组原始圆柱/方柱与 20/50/100 物体房间目录。公开示例带有原始哈希一致的原创 `scene.usda`、布局、冻结规格和来源清单；没有官方机器人资源。映射仅改变读取位置，USD、来源清单与规格绑定检查保持生效。

每条正式试验至少保留 `report.json`、`spec.json`、`layout.json`、`controller_source.json` 和 `trace.jsonl`；在线试验另需 `online_trace.jsonl`。有动态背景时还需 `background_trace.json` 和报告中的初始测量。存在 `supervisor_receipt.json` 时也应选入；它只记录进程退出，不证明任务成功。旧报告内嵌的退出码继续保留，未取得可靠退出记录的试验明确记为未记录，不能补猜。

此外交付当前 `assembly_audit.json`、`assembly_evidence_inventory.json`、审计/构建工具、对应测试、判据源码及其版本摘要。源码打包器自动纳入 `examples/assembly`；`outputs` 证据必须显式选择。运行日志依交付策略排除。缺失最终报告或轨迹的中断试验可以保留其现有输入和控制器，标为证据不足；不要创建伪造报告来补齐。

本地清单检查不等于最终 ZIP 验证。生成归档后需核对 `DELIVERY_MANIFEST.json`、运行包校验器，在解压目录重新运行上述审计，并比较每条试验结论和哈希。源文件或新试验变化后重新生成审计与清单，避免把旧快照当作最终状态。
