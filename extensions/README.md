# Extensions & Archived v1 Modules

本目录存放从 `SceneFactory v2` 核心主干（**图片 + 文字 → 高保真 MuJoCo XML + Random 布局**）彻底解耦拆出的旧版本运行时与动态执行模块，与 `scene_factory/` 核心零耦合：

- `v1_scene_factory/`：v1 时期的 Isaac Sim 后端、Franka 抓杯/恢复控制器、精细装配、JEV 决策适配、HTTP 沙盒运行时、符号规划/执行一致性套件与 Web 后端；
- `v1_web/`：v1 Three.js Web UI 前端；
- `v1_deploy/`：v1 CPU/GPU 容器编排文件；
- `v1_recipes/`、`v1_schemas/`、`v1_examples/`、`v1_tools/`、`v1_tests/`：v1 配方、模式、验收脚本与回归测试。

## 设计边界

- **核心（`scene_factory/`）**：只负责场景生成、MuJoCo XML 编译/片段缓存、物理与视觉裁判、Random 布局变体、原生多模态 Trace 录制、Cordis 插件准入/退出管理及 MCP 插件接口。
- **动态仿真与机器人控制**：属于 VLA / 策略 / 仿真器运行侧（可独立成包或由外部 Agent 自行编排），不再焊死进 SceneFactory 核心主干，也不纳入当前场景构建专利边界。
- 完整的 v1 原始提交历史仍完整保留在 `main` 分支（`bea24ff`）。
