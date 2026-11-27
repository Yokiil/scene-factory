# 文档索引

文档按主题分目录。每篇只记录当时已经核对过的实现与实测，不把未运行的部分写成已完成。

## reference：项目说明与接口

- [PROJECT_OVERVIEW.md](reference/PROJECT_OVERVIEW.md)：项目说明
- [ARCHITECTURE.md](reference/ARCHITECTURE.md)：编译流程与模拟器边界
- [PUBLIC_API.md](reference/PUBLIC_API.md)：受支持的 Python API
- [API_SURFACE_v0.1.md](reference/API_SURFACE_v0.1.md)：v0.1 API 范围
- [CLI.md](reference/CLI.md)：CLI 命令与退出码
- [CLI_REFERENCE.md](reference/CLI_REFERENCE.md)：v0.1 CLI 参考
- [COMPATIBILITY.md](reference/COMPATIBILITY.md)：环境支持范围
- [SCHEMA_POLICY.md](reference/SCHEMA_POLICY.md)：模式兼容性策略
- [SCENE_IR.md](reference/SCENE_IR.md)：通用 Scene IR
- [TESTING.md](reference/TESTING.md)：开发与浏览器验证
- [RELEASE_NOTES_v0.1.0.md](reference/RELEASE_NOTES_v0.1.0.md)：v0.1.0 发布说明

## sandbox：物理沙盒与运行时

- [SANDBOX_COMPLETION_20261104.md](sandbox/SANDBOX_COMPLETION_20261104.md)：2026-11-04 补齐交付，含实测结果与验收边界
- [SANDBOX_DELIVERY_20261028.md](sandbox/SANDBOX_DELIVERY_20261028.md)：2026-10-28 物理沙盒交付说明
- [SANDBOX_PHASE1.md](sandbox/SANDBOX_PHASE1.md)：首阶段实现
- [SANDBOX_BASELINE_20261028.md](sandbox/SANDBOX_BASELINE_20261028.md)：首阶段运行基准记录
- [SANDBOX_SERVICE.md](sandbox/SANDBOX_SERVICE.md)：本地运行服务交付
- [SANDBOX_DEPLOYMENT.md](sandbox/SANDBOX_DEPLOYMENT.md)：单用户沙盒部署与交付包
- [PHYSICS_SANDBOX_PLAN_20261028.md](sandbox/PHYSICS_SANDBOX_PLAN_20261028.md)：优化方案
- [MUJOCO_RUNTIME.md](sandbox/MUJOCO_RUNTIME.md)：MuJoCo 运行、重置与状态恢复
- [MUJOCO_WORKER_POOL.md](sandbox/MUJOCO_WORKER_POOL.md)：MuJoCo 跨会话模型与进程复用
- [RUNTIME_LOCK_DESIGN.md](sandbox/RUNTIME_LOCK_DESIGN.md)：运行环境锁文件 v1 设计草案
- [INCREMENTAL_COMPILER_20261104.md](sandbox/INCREMENTAL_COMPILER_20261104.md)：MuJoCo 增量编译与四组对照交付
- [WEB_UI.md](sandbox/WEB_UI.md)：自然语言可视化界面
- [WEB_RUNTIME_WORKFLOW.md](sandbox/WEB_RUNTIME_WORKFLOW.md)：网页中的 Franka 仿真验收与回放
- [LLM_INTEGRATION.md](sandbox/LLM_INTEGRATION.md)：LLM 接入
- [LOCAL_DEPLOYMENT.md](sandbox/LOCAL_DEPLOYMENT.md)：本地无下载部署

## isaac：Isaac Sim 验收

- [ISAAC_VALIDATION.md](isaac/ISAAC_VALIDATION.md)：Isaac Sim 6.0.1 安装与验收
- [ISAAC_BUILD.md](isaac/ISAAC_BUILD.md)：Isaac 环境包与 HTTP 验收
- [ISAAC_PHYSICS_ESTIMATES.md](isaac/ISAAC_PHYSICS_ESTIMATES.md)：显式物理参数估算
- [ISAAC_ROOM_BACKGROUND_VALIDATION.md](isaac/ISAAC_ROOM_BACKGROUND_VALIDATION.md)：房间背景静置的独立验收
- [EMBODIED_MUG_ACCEPTANCE.md](isaac/EMBODIED_MUG_ACCEPTANCE.md)：Franka 抓杯闭环策略、接触力调节与同回合失抓恢复验收

## assembly：装配、抓持与任务契约

- [ASSEMBLY_RUNTIME_API.md](assembly/ASSEMBLY_RUNTIME_API.md)：装配在线证据接口
- [ASSEMBLY_TRIAL_AUDIT.md](assembly/ASSEMBLY_TRIAL_AUDIT.md)：已录制装配试验审计
- [ASSEMBLY_EVIDENCE_PACKAGING.md](assembly/ASSEMBLY_EVIDENCE_PACKAGING.md)：装配证据的最小交付集合
- [ASSEMBLY_ROOM_FIXTURES_20261104.md](assembly/ASSEMBLY_ROOM_FIXTURES_20261104.md)：20/50/100 对象房间装配夹具
- [CONTACT_GRASP_CONTROL.md](assembly/CONTACT_GRASP_CONTROL.md)：接触反馈抓持控制
- [SE3_POSES_20261104.md](assembly/SE3_POSES_20261104.md)：完整姿态与实例物理参数
- [MANIPULATION_CONTRACTS.md](assembly/MANIPULATION_CONTRACTS.md)：操作契约
- [TASK_CONTRACTS.md](assembly/TASK_CONTRACTS.md)：跨领域任务契约
- [P1_4_TECHNICAL_DESIGN.md](assembly/P1_4_TECHNICAL_DESIGN.md)：P1-4 真实可动部件交互执行设计
- [JEV_ALIGNMENT_20261104.md](assembly/JEV_ALIGNMENT_20261104.md)：JEV 结合方式与验收边界

## assets：资产

- [ASSET_PIPELINE.md](assets/ASSET_PIPELINE.md)：离线 USD 资产接入
- [KITCHEN_DINING_ASSETS.md](assets/KITCHEN_DINING_ASSETS.md)：厨房与餐厅资产包
- [DESSERT_DISPLAY.md](assets/DESSERT_DISPLAY.md)：三层甜点架展示

## data：数据与环境编译

- [DATA_WORKFLOW.md](data/DATA_WORKFLOW.md)：质量审计与场景筛选
- [RUNTIME_DATA_WORKFLOW.md](data/RUNTIME_DATA_WORKFLOW.md)：候选场景的真实运行与证据
- [DECISION_DATASET.md](data/DECISION_DATASET.md)：测量轨迹到决策训练数据
- [ENVIRONMENT_COMPILER.md](data/ENVIRONMENT_COMPILER.md)：环境编译、物理采样与装配契约交付

## patent：专利草稿

- [INCREMENTAL_BUILD_DISCLOSURE_DRAFT.md](patent/INCREMENTAL_BUILD_DISCLOSURE_DRAFT.md)：任务约束与物理依赖图驱动的仿真环境增量构建及验证，技术交底草稿，不是专利申请文件
