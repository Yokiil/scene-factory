# 首阶段运行基准记录

> 本文件保留首阶段代码的历史基线。后续模型复用、HTTP 服务和恢复实测见
> [当前交付说明](SANDBOX_DELIVERY_20261028.md)，不要将两阶段实现混为一组测量。

日期：2026-10-28。以下为最终代码上的本机实测参考，不是加速效果、任务成功率或物理验收。
最后一轮在测试进程结束后运行；开发机操作系统及其他后台负载未隔离。

## 测试条件

- Windows 11 10.0.22631，Python 3.12.7，MuJoCo 3.3.7。
- CPU：Intel64 Family 6 Model 183 Stepping 1，32 个逻辑处理器。
- 主机 GPU 为 RTX 4060 Laptop、驱动 610.47；本测试为 CPU 物理、无渲染，未使用 GPU 吞吐指标。
- `living_room_returned_home`，布局 seed 19，reset seed 0；6 个物体，3 动态/3 静态，
  3 个关节、7 个几何体、0 个网格顶点；运行过程中观测到的最大接触数为 12。
- 单环境、noop 动作；physics dt 0.002 s，action repeat 5，环境步周期 0.01 s。
- MuJoCo solver=2、integrator=3、iterations=100（报告保留引擎枚举原值）。
- 场景文件 SHA-256：`36afced6ec2496c1d146f16721ad19781561342e9e1edcabae2956c2c310e5d8`。

## 实测结果

| 阶段 | 成功样本 | P50 | P95 | 失败 |
| --- | ---: | ---: | ---: | ---: |
| 进程内新建后端至首个观测 | 30 | 5.040 ms | 8.243 ms | 0 |
| 相同场景 reset | 100 | 2.101 ms | 2.417 ms | 0 |
| noop 环境 step | 100 | 0.126 ms | 0.188 ms | 0 |

首次创建与后续创建在同一个 Python 进程内，可能共享导入和系统缓存，不能称为进程冷启动。
重复 reset 仍会重建 MuJoCo 模型，不能称为常驻 worker 温启动或优化后的热重置。
计时包含返回数据的有效性校验，step 结果为纯状态，不包含图像采集与网络开销。

本地原始报告位于 `outputs/phase1_runtime_baseline_20261028/mujoco_state_baseline.json`，
包含全部原始样本、硬件/版本、输入摘要、69 个 Python 源文件摘要及未测项目。
报告生成后已复核这些源码摘要均匹配当前文件。
同目录 `scene/layout.json` 为输入；输出目录不进入版本控制。

复测命令：

```sh
python -m scene_factory.cli build --recipe living_room_returned_home --seed 19 --output outputs/baseline_scene
python -m scene_factory.cli sandbox benchmark outputs/baseline_scene/layout.json --backend mujoco --creation-trials 30 --reset-trials 100 --step-trials 100 --seed 0 --output outputs/mujoco_baseline.json
```

## 验证结果与未完成项

会话 smoke 在同一输入上完成 5 步，重复请求返回相同回执且不再次步进，关闭资源成功。
本地证据：`outputs/sandbox_phase1_20261028/session_smoke.json`。

全量 Python 回归：693 项通过，2 项跳过；发布结构检查未通过。
发布检查剩余原因是本轮开始前已有的四个甜点资产缺少 `source_url`、`source_revision`
及 `THIRD_PARTY_NOTICES.md` 条目。本轮未修改这些资产。新增实现相关的定向测试和 Ruff 通过。
核心依赖检查现使用独立 Python 进程，避免真实 MuJoCo 测试导入 NumPy 后影响检查结果。

本轮未执行 Isaac 抓杯/插接物理验收、Linux GPU smoke、真正冷/温启动、1000 次重置
污染/泄漏测试或恢复误差测试。以上计时不能用于跨后端性能等价比较，也不代表物理稳定性。
