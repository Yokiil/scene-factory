# 方柱房间夹具变体

`room20/room50/room100` 是独立命名的方柱 peg/socket 房间，原创 MIT，包含可直接读取的完整 USD。它们使用 `--peg-shape square --seed 19` 生成，背景参数与原圆柱房间逐项相同。原构建器默认仍为圆柱，原圆柱房间文件与失败证据保留。

```sh
python tools/build_assembly_room_fixture.py outputs/square_room20 --objects 20 --seed 19 --peg-shape square
```

方柱的尺寸、质量、重新计算的惯量和接触边界见相邻 `square_peg_socket/README.md`。原任务规格字节保持一致。`fixture_report.json` 和 `SOURCE.json` 只证明构建输入、非 peg 工作区与背景配置，不能代替机器人装配和背景稳定性验收。动态/静态物体数分别为 7/13、17/33、33/67；基础设施碰撞体另行计数。

审计对动态背景读取 `background_trace.json`，独立重算末尾连续 1 s 的位移、速度和角速度，并要求逐物理步覆盖、对象集合完整且与任务末帧对齐。局部插接通过但背景超限为整体失败，缺失背景证据为未验收。没有由背景静置推出整段任务已稳定的声明。
