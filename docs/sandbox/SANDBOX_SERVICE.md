# 本地运行服务交付

更新：2026-11-04。Isaac USD/Franka 依赖冻结与 HTTP 构建入口见 [Isaac 环境包](../isaac/ISAAC_BUILD.md)。
对应真实 HTTP/GPU 验收须以新报告为准，未通过前不能扩大下文历史 MuJoCo 测试结论。

日期：2026-10-28。提供单用户 HTTP 服务、Python 客户端与 CLI。管理进程不加载仿真器；
每个 session 的独立子进程在 reset/step/observe 之间持续存在。SQLite 保存生命周期、
创建幂等键和逐请求结果/错误证据；服务重启后遗留活动会话标为 `lost`，不会自动重放动作。

## 启动和客户端

在项目 Python 环境中运行，Windows 可使用 `.venv\Scripts\python.exe`：

```sh
python -m scene_factory.cli build --recipe living_room_returned_home --seed 19 --output outputs/baseline_scene
python -m scene_factory.cli sandbox serve --root outputs/runtime-service --scene-root outputs
```

`service serve` 为兼容别名。输入相对路径以 `--scene-root` 为基准；服务数据、状态和
冻结输入写入 `--root`。默认绑定 `127.0.0.1:8765`。非回环地址必须配置 bearer token，
推荐使用 `SCENE_FACTORY_SERVICE_TOKEN` 环境变量，不把 token 写进交付 JSON。
配置 token 后 `/healthz` 同样需要 `Authorization: Bearer ...`。

```python
from scene_factory.runtime_client import RuntimeClient

client = RuntimeClient("http://127.0.0.1:8765")
created = client.create({
    "idempotency_key": "create-1",
    "scene_path": "baseline_scene/layout.json", "backend": "mujoco",
})
sid = created["session_id"]
initial = client.reset(sid, {"idempotency_key": "reset-1", "seed": 42, "options": {}})
request = {
    "episode_id": initial["episode_id"], "sequence_id": 0,
    "idempotency_key": "step-1", "action": {"mode": "noop"}, "steps": 5,
}
result = client.step(sid, request)
assert client.step(sid, request) == result
client.close(sid)
```

独立 CLI 调用同一个持续运行的服务：`sandbox create request.json`、
`sandbox reset SESSION_ID reset.json`、`sandbox step SESSION_ID action.json`、
`sandbox observe SESSION_ID`、`sandbox close SESSION_ID`。各命令支持 `--server-url`、
`--token-env`，并有 `list`、`inspect`、`snapshot`、`restore`、`submit-build`、`build-status`。
请求文件中的字段与 Python/HTTP JSON 相同。创建和构建应由调用者保存并复用幂等键；
Python 客户端省略该键时会为当次调用生成一个新键，但不会自动重试网络请求。

## HTTP API

| 操作 | 路径 | 返回/语义 |
| --- | --- | --- |
| 异步构建 | `POST /v1/builds` | `build_id`；scene/scene_path、registry_path、backend 和幂等键 |
| 查询构建 | `GET /v1/builds/{id}` | queued/running/completed/failed/lost，完成后返回环境包位置及摘要 |
| 创建 | `POST /v1/sessions` | session_id、能力、worker PID、输入摘要和资源限制 |
| 查询 | `GET /v1/sessions`、`GET /v1/sessions/{id}` | 包括已关闭、失败、过期或重启丢失的历史状态 |
| 重置 | `POST /v1/sessions/{id}/reset` | 实际 seed、采样值及初始观测 |
| 步进 | `POST /v1/sessions/{id}/step` | 复用版本化动作契约与幂等语义 |
| 观测 | `GET /v1/sessions/{id}/observation` | 状态；可选 `?modalities=state,rgb&camera=default` |
| 图像产物 | `GET /v1/sessions/{id}/artifacts/{name}` | 会话自有目录下的二进制文件；摘要和时间在观测回执中 |
| 保存 | `POST /v1/sessions/{id}/snapshots` | `{idempotency_key, level}`；`rebuild` 支持 dry_run/MuJoCo，`runtime_state` 仅支持 MuJoCo；Isaac 不支持 |
| 恢复 | `POST /v1/sessions/{id}/restore` | `{idempotency_key, snapshot: 保存回执}`，受后端能力约束 |
| 释放 | `DELETE /v1/sessions/{id}` | 关闭资源；重复释放仍返回终态 |

创建请求必须在 `scene`、`scene_path`、`environment_path` 中选择一个。
普通场景会冻结布局和 registry；资产闭包标记为 `not_locked`。
`environment_path` 支持 MuJoCo 和 Isaac 环境包：先验证、复制声明的依赖闭包、再次验证，
然后从该会话副本加载。该模式禁止 registry/USD/backend_options 覆盖，避免偏离运行锁。
异步构建为单个活动构建进程、最多 16 个排队/运行作业；300 秒超时会回收构建进程。

默认限制：2 个活动会话、30 秒请求超时、60 秒 worker 启动超时、300 秒空闲租约、
4 MiB JSON 请求体。可设置线程/内存的总准入预算和会话资源请求；这些是准入规则，
不是显存或 RAM 硬配额。容器 CPU/RAM/GPU 边界由部署平台配置。
租约回收不会抢占正在执行的请求；超时请求会终止 worker，保留 failed 证据并释放容量。

创建请求可指定 `history_limit`，整数范围 1–100,000，默认 4,096。该参数限制本会话保存的
幂等请求历史条目数；长轨迹可显式使用 `"history_limit": 10000`。已使用的键不会因达到预算
而自动淘汰；旧键重试仍可返回原回执，新请求超出预算时明确拒绝。step 历史在成功 reset 时
清理；reset 历史按会话累计。较高预算会增加 RAM 消耗，条目数边界不构成内存字节硬配额，
调用者仍需配置合适的资源请求与容器限制。MuJoCo 池复用时新会话使用自己的历史预算，
不会继承旧会话的条目或提高原会话预算。

可用 `--mujoco-worker-pool` 启用严格兼容的跨会话 MuJoCo 模型/进程复用；默认仍关闭。
idle worker 计入 CPU、内存及驻留进程准入预算；不匹配的 idle 会先销毁以腾出容量。
每个新会话的 episode、状态、幂等历史和图像目录独立。异常污染、超时及租约过期的
worker 不回池；Isaac 不池化。预热、TTL、最大复用次数和实测说明见 `MUJOCO_WORKER_POOL.md`。

未知字段、越界路径、非有限 JSON、重复 JSON 键、超大请求和浏览器跨源请求拒绝。
输入目录与产物目录有路径边界，产物只能按单个文件名读取。
400 表示请求错误，409 表示状态/能力/容量冲突，410 表示已失效的 worker，
503 表示正在关闭或会话忙，504 表示 worker 超时。

## 后端与恢复边界

MuJoCo 的状态快照可经 HTTP 保存/恢复；恢复等级和误差解释以会话能力与后端记录为准。
`rebuild` 表示重建初始化，不表示恢复任意接触瞬间。图像能力按实际后端检查；
不能因为 HTTP 可取文件，就认定机器人任务或物理装配通过。

Isaac 必须使用 Isaac Python 启动服务，并指定受允许目录约束的 `usd_path`；
如需机器人资产，可配置 `backend_options.robot_asset_root`。Windows ASCII 路径别名在
通过真实路径边界检查后保留。RGB-D 需要 `backend_options.enable_rgbd=true`。
首次引擎 reset 可能较慢，应按机器调整 `--request-timeout`。

现有 Isaac `SimulationApp.close()` 会使 worker 正常退出。服务把显式关闭时的退出码 0
识别为成功回收；每个 Isaac worker 暂时仅允许一个 episode，重试原 reset 键仍可用。
Isaac 能力表的 `snapshot_levels=[]`：当前不支持导出快照，也不支持恢复快照。
新 episode 需创建新会话，并使用调用者另行保存的初始 seed/options；这不构成快照恢复。
当前 worker 内的第二次 reset/restore 明确拒绝，并在 `service_constraints` 中声明此限制。
一次只准入一个 Isaac worker。

## 已运行验证

`tests/test_runtime_service.py` 覆盖真实子进程和 HTTP 往返、并发幂等步进、创建键冲突、
输入冻结、超时终止、崩溃/重启状态、空闲租约、目录锁、锁文件构建和会话加载。

```sh
python tools/smoke_runtime_service.py --output outputs/delivery_20261028/http_service
```

本机 MuJoCo smoke 已通过：5 步、相同请求回执、同一个 worker PID、运行状态恢复后的
时间和物体位置一致、关闭与重复关闭成功。证据位于输出目录的 `service_smoke.json`。
这是本机协议和恢复测试。2026-11-04 另有本机真实 Isaac HTTP 构建、外部关节目标与
RGB-D 对齐验收，记录在 `outputs/completion_20261104/isaac_http03/report.json`，入口见
[ISAAC_BUILD.md](../isaac/ISAAC_BUILD.md)。这项短程协议验收不包含装配任务成功；Linux/NVIDIA
服务器和共享多用户访问仍未验收。
