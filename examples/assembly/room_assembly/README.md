# 房间中的原始 peg/socket 工作台

这三档场景在原始四对象工作区外增加背景，完整保留机器人安装座、工作台、插销、插座
的 USD 属性和冻结装配阈值。背景包含动态刚体和静态碰撞体，动态背景的五面固定围挡
仅在图像中隐藏，物理碰撞仍启用。`SOURCE.json` 记录全部参数、计数和工作区保留校验。

使用提供 `pxr` 的 Python 生成，无须启动 Isaac 或使用 GPU：

```sh
python tools/build_assembly_room_fixture.py outputs/room20 --objects 20 --seed 19
python tools/build_assembly_room_fixture.py outputs/room50 --objects 50 --seed 19
python tools/build_assembly_room_fixture.py outputs/room100 --objects 100 --seed 19
```

`fixture_spec.json` 描述固定配方；生成器参数控制数量和 seed。随附的 `room20/50/100`
文件为 seed 19 的原始几何。所有生成几何均为项目原创 MIT，未包括外部机器人资产。

**构建成功不是动力学或装配任务通过。** 请在对应引擎中记录背景静置、实际物体/接触数量、
机器人任务结果和资源；先前桌面任务结果不能自动扩展到这些房间。
