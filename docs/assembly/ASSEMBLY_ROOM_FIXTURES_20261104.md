# 20/50/100 对象房间装配夹具

日期：2026-11-04。`tools/build_assembly_room_fixture.py` 使用原始
`build_assembly_fixture.py` 先构建工作区，再用 OpenUSD 扩展背景；不启动仿真，不使用 GPU。

四个原始工作区对象及原接触材料、物理场景的 Sdf 规范在扩展前后做规范化 SHA-256 比较；
原始对象布局、机器人任务配置和装配规格保持一致，冻结阈值没有变化。房间地板扩展到
14×14 米，增加四周固定碰撞墙。所有资源无外部几何依赖，机器人仍由运行时加载。

| 场景对象数 | 动态对象 | 静态对象 | USD 碰撞 prim | 背景围挡碰撞 prim |
| --- | ---: | ---: | ---: | ---: |
| 20 | 7 | 13 | 59 | 30 |
| 50 | 17 | 33 | 139 | 80 |
| 100 | 33 | 67 | 269 | 160 |

对象数包含原始四对象，不把环境地板、四周墙和每个动态背景的五面固定围挡混算成物体。
因此对象数量和碰撞形状数量必须分别报告。背景尺寸、密度推导质量/惯量、摩擦、零初始
速度、seed、碰撞接触偏置及围挡参数均在各夹具 `SOURCE.json` 或 USD 中显式记录。

背景本体与围挡的完整 AABB 均避开 `[-1.5,-1.5,0]` 至 `[2,1.5,2.3]` 米的保守机器人、
抓取、插入和撤离预留体积。动态背景从地板上方 2 mm 初始化并置于固定围挡内；这不等于
实测已稳定，仍须检查初始沉降、求解器异常和背景运行后状态。

原始几何和配方在 `examples/assembly/room_assembly/`。ASCII 运行路径（`<runtime-root>` 为本机运行根目录，原始记录见 `examples/assembly/fixture_path_mappings.json`）为：

```text
<runtime-root>/completion_20261104/room20
<runtime-root>/completion_20261104/room50
<runtime-root>/completion_20261104/room100
```

每个目录包含 `scene.usda`、`layout.json`、原始 `assembly_spec.json`、`SOURCE.json`
和 `fixture_report.json`。`SOURCE.json` 的 `geometry` 字段保留原始插接夹具格式，原始
控制器可按相同输入契约运行。另用 `room_fixture` 字段标注背景规模及当前未评估状态。

此处只验收原创几何构建、输入确定性、工作区隔离和冻结契约保留。机器人装配、背景
稳定性、实际接触和资源指标应由后续运行报告给出；不能用桌面成功报告替代房间任务报告。
