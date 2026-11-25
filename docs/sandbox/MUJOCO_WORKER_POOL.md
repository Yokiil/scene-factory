# MuJoCo 跨会话模型与进程复用

2026-11-04 新增可选 MuJoCo worker 池，默认关闭。复用对象包括已经编译的 `MjModel` 和所属子进程；每个新会话仍创建新的 `SceneFactoryEnv`、`RuntimeSession`、episode、状态版本、请求历史和产物目录。初版要求原始场景、生成的完整 MJCF、registry、相关源码、引擎二进制及版本、Python、后端设置和资源请求全部兼容。场景或求解器参数变化会使用新 worker；不复用 Isaac 或 dry-run worker。

```powershell
.venv\Scripts\python.exe -m scene_factory.cli sandbox serve --root outputs\server --scene-root outputs --mujoco-worker-pool --pool-max-idle 1 --pool-idle-seconds 60 --pool-max-uses 100
```

`--max-sessions` 同时限制可驻留的 active + idle worker 总数；活动会话上限仍受它约束。空闲 worker 消耗同样的 CPU 线程与内存准入预算。新请求需要容量时先销毁不匹配的 idle worker，然后启动新进程。准入预算不构成 Windows 或 Linux 进程 RAM/CPU 硬限。

关闭健康会话时，worker 检查完整积分状态的有限性和 solver warning；丢弃 episode 数据、任务计数器及 renderer，只保留编译模型。新会话第一次 reset 按原有 reset 路径创建全新 `MjData`，重新初始化状态并更新初始位姿。如果上一 episode 的随机化改变了几何或物理参数，后端会按其结构比较规则重新编译；不能仅凭进程复用宣称模型命中。

任何 worker 请求异常都会永久标记该进程不可再入池；即使随后会话 reset 成功也不会清除该标记。仿真非有限状态、solver warning、进程死亡、请求超时、会话租约过期、idle 超时和达到最大使用次数都会销毁进程。未 reset、尚未编译模型的空 Python worker 不进入池。服务退出会回收 active 和 idle 进程。服务重启不会恢复旧进程。

`DELETE /v1/sessions/{id}` 返回 `worker_disposition=pooled|destroyed`。旧会话立即进入终态，新会话不会继承其 reset/step 幂等键。创建回执包含 `worker_reused` 和 `worker_pool_compatibility_sha256`；会话描述中的 `worker_runtime` 记录 PID、复用次数、编译计数和污染标记。`GET /v1/sessions` 与 `/healthz` 提供 worker 池计数。相关 Python 文件在部署期间应保持不可变；更新源码或运行环境后重启服务。

可通过公开 API 预热一个场景并关闭会话，将其 worker 放入池：

```powershell
.venv\Scripts\python.exe tools\prewarm_mujoco_pool.py create-request.json --server-url http://127.0.0.1:8765 --seed 0
```

预热器返回 `parked` 才表示服务保留了 worker。后续请求需具有相同配置和资源请求，且在 idle TTL 内到达。

验证工具：

```powershell
.venv\Scripts\python.exe -m pytest tests\test_worker_pool.py -q -o addopts=""
.venv\Scripts\python.exe tools\validate_worker_pool.py --output outputs\worker_pool_validation --trials 30 --reset-trials 1000
```

工具分别记录至少 30 次新进程创建和 30 次池复用创建到首个有效观测的时间；池的首次填充单列，不混入 warm 统计。报告保留每次 PID、会话/episode、编译计数、兼容摘要、P50/P95、源码版本和条件。额外进行 1,000 次外力扰动后 reset，比较位置、完整姿态及速度，并采样 worker 内存。该结果只是本机场景及状态观测的有界测试，不是跨硬件稳定性、任意模型无泄漏或机械臂任务验收。

本机当前源码证据：`outputs/completion_20261104/worker_pool_release_final/report.json`。
该组在修复“启动时间消耗初始空闲租约”后重跑；此前各组保留为历史记录。
两个创建组各 30 次，新 worker 到初始状态观测的 P50/P95 为 0.7615/0.8700 秒；
池复用为 0.1381/0.1633 秒，P50 降低 81.87%。30 次温创建保留同一 PID，模型总编译次数为 1；
1,000 次扰动后 reset 的位置、姿态及速度最大分量差为 0，HTTP reset P50/P95 为
12.489/33.567 ms。采样 private bytes 从 56,545,280 到 56,549,376，working set 增加
126,976 bytes；这不构成一般性的无内存泄漏证明。场景只有一个动态代理物体，
`dt=0.002s`、`frame_skip=1`、无渲染；管理进程和 OS 缓存常驻，未清空宿主文件缓存，
OS 背景负载未隔离。报告中的 `source_unchanged=true` 表示测量期间相关源码未变。
