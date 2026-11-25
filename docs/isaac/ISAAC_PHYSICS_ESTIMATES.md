# Isaac 的显式物理参数估算

日期：2026-11-04。此功能用于本轮精细接触试验，默认关闭，不修改共享机器人资产。

本机 Franka 资产的 PhysX 读回显示，手指惯量比同质量、同碰撞几何尺寸的均匀体估算小约一万倍。
这一发现绑定本地资产与引擎版本，不代表所有官方 Franka 资产均有相同问题。估算惯量也不是
真实机器人 CAD 或实测惯量，不能据此宣称完成 sim-to-real 标定。

在运行布局的 `task.physics_overrides` 中显式提供
`scene_factory.robot_physics_overrides.v1` 配置。完整示例在交付证据
`outputs/completion_20261104/corrected_physics_config.json`，含来源、估算方法和
`real_world_measured: false`。配置按原始字节摘要记录，进入锁定环境包的场景输入。
MuJoCo 后端与其构建器明确拒绝此 Isaac 专用配置。

功能行为：

- 保留各机器人 link 的原质量，要求启用且可读取的凸碰撞网格。
- 在匿名 stage session layer 请求 PhysX 按均匀密度碰撞体重算惯量、质心及主轴。
- 只在该 stage 中解除必要的几何实例化，应用显式 contact/rest offsets；原 USD 层和本地资产缓存不写回。
- 可指定 articulation、任务动态体和 scene 的求解迭代设置；初始化后读回，若上下文改写则拒绝。
- 保存原值、几何摘要、生效配置和原始 PhysX 张量。校验质量、有限性、正定性、惯量三角不等式、
  尺度与质心范围；参数未生效时初始化失败。

PhysX 的张量为 FP32。校验非对称残差时使用
`8 * eps(float32) * max(abs(tensor))`，原始矩阵保持不变，不对称化也不写回。
该数值精度处理不改变装配速度、力、位置或角度门槛。

精细插接试验可显式使用该变体：

```sh
python tools/run_precision_assembly.py <fixture> --output <new-output> \
  --asset-root <local-franka-assets> --online \
  --physics-overrides outputs/completion_20261104/corrected_physics_config.json
```

应使用 Isaac 专用 Python，并使用新的输出目录。该工具只发送关节驱动目标；参数估算不授予
直接移动零件或重写速度的权限。实际装配结论另见各试验报告及 `assembly_audit.json`，
参数读回通过不能代替任务通过。

原始诊断、修正读回、浮点残差和离线复核分别保存在 `diagnostics/`。
完整运行接口见 [Isaac 环境包](ISAAC_BUILD.md) 与 [在线装配](../assembly/ASSEMBLY_RUNTIME_API.md)。
