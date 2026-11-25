# Isaac 房间背景静置的独立验收

`tools/validate_isaac_room_background.py` 每个进程只验证一个房间。阶段标记固定为
`background_settling_only_not_robot_task`。机器人通过 `RuntimeSession` 的
`joint_position` 接口保持 reset 后实测的初始七关节目标和夹爪开度；初始化之后只提交关节驱动目标。
不执行抓取、搬运或插接，不据此判定装配任务成功。

```powershell
& $IsaacPython tools/validate_isaac_room_background.py $RuntimeRoot/completion_20261104/square_room20 --asset-root $AssetRoot --duration-s 5 --physics-dt-s 0.008333333333333333 --output outputs/completion_20261104/isaac_background_room20
```

将 fixture 和 output 分别改为 `square_room50`、`square_room100` 对应路径运行另外两档。
必须串行运行；每次使用新输出目录。也可直接使用公开的
`examples/assembly/square_room_assembly/room20|50|100`，运行时仍需取得兼容的官方机器人资产。
这些公开 fixture 的原始 USD、布局、规格和 SOURCE 必须与记录哈希一致。

可选 `--physics-overrides 配置.json` 复用核心的显式 `task.physics_overrides` 配置。
它只修改本次运行布局与匿名 stage session layer，保留原始 fixture 字节，记录输入配置哈希、
规范化配置、运行配置及实际 readback。几何估计须按配置声明为非实测值。
工具不会修改验收阈值或原装配规格，也不会把配置被接受等同于物理验证通过。

`trace.jsonl` 保存 reset 后初始观测以及每个实际完成的物理步：所有背景的完整世界位姿、
线速度、角速度，机器人关节状态，原动作、状态版本、仿真时刻，以及全场接触对和覆盖证明。
动态背景必须由有效 PhysX 句柄读取原始运动；静态背景的位姿来自 USD，速度按静态定义为零，
两种来源在 trace 内明确标注。接触数据为各对象对逐接触点法向力绝对值之和、合力幅度及点数；
不宣称保存逐点向量、摩擦向量或软体接触。

每一行均检查全部场景对象和机器人处于接触覆盖范围，碰撞形状及缓冲区清单完整、未饱和，
法向力单位与时间窗正确。漏采样、错 episode、状态序号不连续、缺失姿态或原始速度均不能通过。
原始动作和接触覆盖可能使观测较大，因此这不是纯 PhysX 内核吞吐测试。

最终连续至少 1 秒的每步窗口沿用既定背景门槛：相对窗口首帧的最大位置漂移 ≤ 1 mm，
末帧相对 reset 后初始观测的总位移 ≤ 5 mm，窗口内原始线速度 ≤ 5 mm/s，
原始角速度 ≤ 0.05 rad/s。初始参考已经经过引擎 reset 的预热步骤；不声称覆盖预热过程。
全程轨迹仍完整保存，阈值的判断范围明确限定在末尾窗口。

`timings.jsonl` 逐次保存真实 `RuntimeSession.step(steps=1)` 的墙钟耗时。
P50/P95 仅使用实际成功返回的调用，另报失败调用数、执行步数和实际仿真时长。
单步耗时包含接触观测、协议包装、重试历史维护；不包含 JSONL 写盘、启动、reset 或渲染。
这里只报告一次运行的样本量与描述性统计，不冒充 30 次冷/温启动或独立重复实验。

报告冻结四个原始 fixture 文件、运行布局、全部 `scene_factory` Python 源码及本验证器哈希；
结束时重新核对原始输入与源码未变。`output_sha256` 绑定生成文件，`report.sha256` 单独绑定报告。
`background_acceptance` 仅在请求步数全部完成、全程观测覆盖完整、输入源码未变且最终窗口达标时为
`passed`。缺证据为 `not_verified`，完整测量下越限为 `failed`；`assembly_acceptance` 始终为
`not_evaluated`。

Isaac 的 fast shutdown 可能在 `session.close()` 内直接退出解释器，因此报告和哈希在关闭前写入，
`cleanup` 为 `pending_process_shutdown`。外部监督进程应另存退出码与该报告哈希的 receipt。
退出码 0 本身不代表背景或机器人任务验收通过。Linux GPU 部署仍需独立目标服务器验收。
