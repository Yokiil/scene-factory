# 单用户沙盒部署与交付包

初版日期：2026-10-28；更新：2026-11-04。用户当前没有 Linux GPU 服务器，本次交付部署文件与检查脚本，
**Linux 容器构建、GPU 驱动/引擎兼容和服务器物理任务仍未验收**。
本机 Docker CLI 能解析 Compose 配置，但 Docker Desktop Linux engine 未启动，未运行容器。

## 部署内容

| 文件 | 用途 |
| --- | --- |
| `deploy/sandbox/Dockerfile.mujoco` | Python 3.12.7、MuJoCo 3.3.7 CPU 服务镜像 |
| `deploy/sandbox/requirements-mujoco.txt` | 与已测开发环境对应的精确 Python 依赖版本 |
| `deploy/sandbox/Dockerfile.isaac` | 基于用户指定的真实 Isaac Linux 镜像摘要构建 |
| `deploy/sandbox/compose.yaml` | CPU/GPU 独立 profile、资源限制、资产/结果挂载、健康检查 |
| `deploy/sandbox/probe.py` | 分开检查 Linux 宿主与容器内依赖 |
| `deploy/sandbox/headless_smoke.py` | 通过 HTTP 创建、重置、步进、检查重试和释放会话 |

Python 依赖版本已经固定；Linux wheel 的获取与运行仍需目标环境验证。
CPU 基础镜像默认使用明确版本 tag，可用 `PYTHON_BASE_IMAGE` 改为实际 `image@sha256:...`。
构建后应记录最终 OCI image ID、基础镜像 digest、宿主驱动、GPU、引擎及 solver 信息。
Debian 包仓库安装结果由最终镜像摘要锁定，不把一个版本 tag 当作不可变镜像证明。

两个 Dockerfile 使用单独的 Docker ignore 白名单，构建上下文不包含 `.env`、
`config/llm.json`、虚拟环境、Git 历史、模型缓存和本地运行日志。

## CPU 快速启动（在目标 Linux 主机上）

以下命令从交付包解压后的项目根目录运行，要求 Docker Engine、Compose v2 和 Python 3.12。

```sh
mkdir -p outputs/sandbox_server/scenes outputs/sandbox_server/runtime
python3 -m scene_factory.cli build --recipe living_room_returned_home --seed 19 --output outputs/sandbox_server/scenes/living_room
sudo chown -R 10001:10001 outputs/sandbox_server/runtime
export SCENE_FACTORY_SERVICE_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
python3 deploy/sandbox/probe.py --stage host --backend mujoco --assets data/assets --output outputs/linux_host_probe.json
docker compose -f deploy/sandbox/compose.yaml --profile cpu build sandbox-cpu
docker compose -f deploy/sandbox/compose.yaml --profile cpu up -d sandbox-cpu
docker compose -f deploy/sandbox/compose.yaml exec sandbox-cpu python deploy/sandbox/probe.py --stage container --backend mujoco --assets /opt/scene-factory/data/assets --output /data/runtime/container_probe.json
python3 deploy/sandbox/headless_smoke.py --url http://127.0.0.1:8765 --scene /data/scenes/living_room/layout.json --backend mujoco --steps 5 --output outputs/linux_cpu_smoke.json
```

token 仅用于本实例的 API 访问，容器未配置 token 会拒绝启动。不要把生成的值写入交付源码。
Compose 默认只把端口发布到宿主 `127.0.0.1`；远程使用可通过 SSH 隧道转发。
这是一项单用户服务，不能据此宣称已具备多租户权限与隔离。

CPU profile 使用 UID/GID 10001，4 CPU、8 GiB 内存、256 进程上限，容器文件系统只读，
允许写入挂载的运行目录和 `/tmp`。服务接纳最多 2 个会话，内存接纳预算 6144 MiB、CPU 预算 4；
接纳预算不替代容器对整个进程组的硬限制。

## GPU / Isaac 配置

未虚构一个“可用的 Isaac 6.0 Linux 镜像 tag”。`ISAAC_BASE_IMAGE` 必须由目标环境提供
且使用 `@sha256:` 的真实 64 位摘要。Dockerfile 会检查 Python >=3.12 和
`EXPECTED_ISAAC_VERSION`（默认对应本机检测到的 `6.0.1.0`）；不匹配则构建失败。
若供应商镜像的 Python 入口不同，通过 `ISAAC_PYTHON` 指定。所有镜像和资源许可由部署者按
实际取得的发行版配置，Compose 不自动同意 Isaac EULA。

准备好镜像、NVIDIA Container Toolkit、可用官方 Franka 资产和 USD 依赖后：

```sh
export SCENE_FACTORY_SERVICE_TOKEN="${SCENE_FACTORY_SERVICE_TOKEN:-$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')}"
python3 deploy/sandbox/probe.py --stage host --backend isaac --assets data/assets --output outputs/linux_gpu_host_probe.json
# 按实际镜像设置 ISAAC_BASE_IMAGE=image@sha256:<真实摘要>
# 按该发行版的要求设置 ISAAC_ACCEPT_EULA、ISAAC_PRIVACY_CONSENT
mkdir -p outputs/sandbox_server/gpu-runtime
docker compose -f deploy/sandbox/compose.yaml --profile gpu build sandbox-gpu
docker compose -f deploy/sandbox/compose.yaml --profile gpu up -d sandbox-gpu
docker compose -f deploy/sandbox/compose.yaml exec sandbox-gpu /usr/local/bin/scenefactory-python deploy/sandbox/probe.py --stage container --backend isaac --assets /opt/scene-factory/data/assets --output /data/runtime/container_probe.json
python3 deploy/sandbox/headless_smoke.py --url http://127.0.0.1:8766 --scene /data/scenes/franka/layout.json --usd /data/scenes/franka/scene.usd --backend isaac --steps 5 --output outputs/linux_isaac_smoke.json
```

最后一条命令依赖已放入 `outputs/sandbox_server/scenes/franka` 的真实参考场景及完整资产依赖。
它使用引擎默认的 Franka 资产解析器；`headless_smoke.py` 没有 `--robot-asset-root` 参数。
需要指定本地机器人资产根时，先按 [Isaac 构建接口](../isaac/ISAAC_BUILD.md) 调用 `sandbox submit-build`，
在请求 JSON 顶层提供容器内的 `robot_asset_root`，再用返回的 `package_dir` 创建
`environment_path` 会话。本地资产根须含 `Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd`，
并位于服务允许读取的挂载目录中；原始宿主绝对路径不会自动变成容器路径。
预热时应提前解析所有外部 USD 依赖，不能仅复制根 USD。预检读取引擎和 CUDA 信息，
不会启动 `SimulationApp`；HTTP smoke 才会实际创建引擎。5 步 smoke 只验收启动和协议，
不能代替抓杯成功、RGB-D 对齐或插接接触判据验收。

GPU profile 默认设备 0、单会话、8 CPU、16 GiB RAM、2 GiB shared memory。
`NVIDIA_VISIBLE_DEVICES` 可改为实际单设备编号。显存没有硬配额承诺；需在目标驱动和
部署平台上测量峰值显存并验证接纳数。驱动在宿主检查，CUDA/引擎在容器内检查，资产挂载
单独检查。CPU profile 的成功不能用于替代 GPU 验收。

## 生命周期与恢复

`/data/runtime` 持久化 SQLite 作业/会话状态、事件证据、冻结输入、图像和快照产物。
每个活动会话有独立常驻 worker，空闲租约超时、调用超时或进程崩溃会触发回收。
空闲租约从 worker 初始化完成并注册为 ready 后开始，启动阶段单独受 startup timeout 约束。
服务重启后把原活动会话标为 lost，而不是假装已恢复 GPU/仿真进程。

CPU profile 可设置 `SCENE_FACTORY_MUJOCO_POOL=1` 启用严格兼容的 MuJoCo 模型/进程复用；
默认 `0`。`SCENE_FACTORY_POOL_MAX_IDLE=1`、`SCENE_FACTORY_POOL_IDLE_SECONDS=60`、
`SCENE_FACTORY_POOL_MAX_USES=100` 控制空闲容量、TTL 和退休阈值。idle worker 计入准入预算，
进程资源硬限制仍由容器配置提供。GPU profile 不启用该环境变量，Isaac 不池化。
详见 `docs/MUJOCO_WORKER_POOL.md`。上述容器开关已交付；Linux 容器执行仍未验收。

```sh
docker compose -f deploy/sandbox/compose.yaml logs --tail 100 sandbox-cpu
docker compose -f deploy/sandbox/compose.yaml stop sandbox-cpu
docker image inspect scene-factory-sandbox:mujoco-local --format '{{.Id}}'
```

排障先读 probe 与会话失败证据，再确认资产路径、版本和容器资源。旧会话不能靠复用 ID
恢复；创建兼容新会话并显式使用受支持的快照恢复协议。删除容器不会删除 bind mount 的证据，
`down -v` 会删除命名缓存卷，因此不要把它当作日常重启步骤。

## 源码与证据交付

`tools/package_sandbox_delivery.py` 使用 Git 已跟踪清单加明确源码目录，包含尚未提交的新实现。
它排除密钥/环境配置、本地资源缓存、依赖目录、数据库、日志和未选中的 outputs。
可显式加入 `outputs/delivery_20261028` 或 `outputs/completion_20261104` 下的证据文件或目录；不把整个工作区打包。
每个文件有大小与 SHA-256，ZIP 内含 `DELIVERY_MANIFEST.json`，外部附 ZIP 的 `.sha256`。
校验器拒绝路径越界、符号链接、重复文件、缺文件、未列明内容及哈希不匹配。

打包会检查注册表与 `SOURCE.json` 的文件引用，解析 glTF/GLB 外部资源，使用 OpenUSD
解析全部 USD 层的间接依赖。缺文件、远端未解析资源或超出 `data/assets` 的依赖会使打包失败。
运行打包的 Python 必须可导入 `pxr`，也可用 `--usd-python /path/to/vendor/python`
指定另一个具备 OpenUSD 的 Python 解释器。该参数需要解释器可执行文件，不是 shell 启动脚本。
公开机器人/相机配置目录 `config/robots`、`config/cameras`、`configs/robots`、
`configs/cameras` 会被纳入；`config/llm.json` 和凭据始终排除。

```sh
python3 tools/package_sandbox_delivery.py create --output deliveries/SceneFactory-sandbox-20261104.zip --evidence outputs/completion_20261104/environment_final
python3 tools/package_sandbox_delivery.py verify deliveries/SceneFactory-sandbox-20261104.zip
delivery_sha="$(cut -d ' ' -f1 deliveries/SceneFactory-sandbox-20261104.zip.sha256)"
python3 tools/validate_sandbox_archive.py deliveries/SceneFactory-sandbox-20261104.zip --output outputs/archive_smoke --expected-sha256 "$delivery_sha"
```

解包验收要求新的输出目录；它从解包后的源码启动真实 MuJoCo HTTP 会话，运行构建、外力步进、
幂等、恢复与兼容模型复用。Python 解释器仍须提供 MuJoCo 等依赖，验收不会自动安装依赖。
通用源码打包器拒绝选择含私有 Isaac 机器人资产闭包的目录或 ZIP；选择原始报告/轨迹，
将相应本地环境包保留在资产持有者的运行目录中。

打包包含源码和选定证据，不包含商业引擎、GPU 驱动、外部模型或许可证受限资源的隐式副本。
交付包可核验不等于整个研究方案的所有物理验收已通过；实际状态以随包验收矩阵和原始证据为准。
