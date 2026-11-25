# 独立方柱插接夹具

`original_square_peg_socket_v1` 是原圆柱插接案例的独立几何变体，原创 MIT。20×20×100 mm 方柱与原 21.5 mm 方孔在轴对齐时每侧间隙为 0.75 mm，质量保持 0.08 kg。方柱体积、密度和惯量随形状重新计算；`SOURCE.json` 记录原/新 USD、构建器和冻结规格摘要。

原 `assembly_spec.json` 的字节、所有数值门槛、底座/工作台/插孔、零件初始位置和机器人基座保持一致。方柱的角点、偏航容纳范围和平面夹持接触与圆柱不同，新变体的物理结果不能替代原圆柱失败记录。这里的 `fixture_report.json` 仅证明 USD 构建和不变输入检查，任务验收为 `not_verified`。

以下命令只需要提供 OpenUSD 的 Python，不启动 GPU：

```sh
python tools/build_square_assembly_fixture.py outputs/square_fixture
```

运行机器人试验时仍需另行取得并配置官方 Franka 资产。源码示例不包含它。解压后的审计可以将保存报告内原夹具路径映射到此目录，哈希检查保持生效：

```sh
python tools/audit_assembly_trials.py <trial-directory> --fixture-map-file examples/assembly/fixture_path_mappings.json --output outputs/square_audit.json
```
