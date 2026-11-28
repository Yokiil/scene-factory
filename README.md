# SceneFactory (v2)

> **图片 + 文字 → 高保真、低成本、可随机化的 MuJoCo XML，以及可直接用于后训练的原生 Agent Trace。**

SceneFactory v2 从原先大而全的集成式框架彻底瘦身为**极简、独立、AI 友好的 Scene-to-MJCF 引擎**：

- **单一清晰目标**：接收自然语言与可选参考图片，通过多轮 VLM/LLM + MuJoCo 客观物理/视觉反馈闭环，快速产出高保真 `scene.xml` 与确定性 Random 布局变体；
- **场景与仿真执行彻底解耦**：像 Docker 构建镜像一样只负责产出干净可跑的环境 XML，通过 CLI / Python / 标准 MCP 插件提供给外部 Agent；旧版动态机器人控制、Isaac 机械臂验收、精细装配与 Web 队列全部隔离在 [`extensions/`](extensions/README.md)（完整 v1 保留在 `main` 分支）；
- **真 Agent，而非 Workflow**：不焊死 bad-case 拦截代码或硬枚举状态机，假设 LLM 具备规划与自纠能力，并将完整的多模态推理、工具调用、渲染图自查与物理裁判分数记录为可直接转 SFT/RL 的 `trace.json`；
- **Cordis 插件治理约束准入与退出**：所有提示注入与技术护栏均通过 [`config/cordis.toml`](config/cordis.toml) 以可逆插件挂载，支持一键退出或 `--no-plugins` 跑裸模型基线，模型升级即可直接吃上红利。

详细架构见 [`docs/ARCHITECTURE_V2.md`](docs/ARCHITECTURE_V2.md)。

---

## 快速安装

核心零第三方运行时依赖（Python >= 3.12）；运行真实 MuJoCo 物理验证、离屏渲染与快速原地随机化需安装 `mujoco` 扩展：

```bash
python -m pip install -e ".[dev,mujoco]"
```

---

## 快速上手

### 1. 图片 + 文字生成高保真 MuJoCo XML（含 Random 布局与 Trace）

配置 OpenAI-compatible 多模态模型端点（环境变量 `SCENE_FACTORY_LLM_BASE_URL`、`SCENE_FACTORY_LLM_MODEL`、`SCENE_FACTORY_LLM_API_KEY` 或 `config/llm.json`）：

```bash
scene-factory generate \
  --prompt "一张木桌，上面放一个蓝色马克杯和一个碗，整体干净整齐" \
  --image reference.png \
  --randomize 10 \
  --output outputs/table_mug_scene
```

关闭全部 Cordis 引导/护栏插件，直接评测裸模型能力：

```bash
scene-factory generate --prompt "..." --no-plugins --output outputs/naked_baseline
```

输出目录包含：`scene.xml`、`variants/variant_*.xml`、`renders/verify_*/cam_*.png` 与可转后训练的 `trace.json`。

### 2. 独立校验、随机化、查资产与导出 SFT Trace（无需 LLM）

```bash
# 搜索本地 51 条资产库（含多部分凸包高保真网格）
scene-factory assets --query "杯 桌 碗"

# 客观编译、静置仿真并离屏渲染任意 MuJoCo XML
scene-factory verify outputs/table_mug_scene/scene.xml --render-dir outputs/renders

# 对任意 MuJoCo XML 做快速确定性 Random 布局与物理参数变体（原地 MjModel patch，免重复编译）
scene-factory randomize outputs/table_mug_scene/scene.xml \
  --count 20 --seed 42 --xy-jitter 0.08 --yaw-jitter 30 --mass-scale 0.85 1.15 \
  --output outputs/random_variants

# 查看当前 Cordis 插件准入与退出标准
scene-factory plugins

# 将物理裁判达标的自主纠错 Trace 导出为 SFT / RL JSONL
scene-factory export-sft outputs/ --min-score 0.8 --output outputs/sft_traces.jsonl
```

### 3. 启动 Precision 3D Studio 交互式工作台（零前端构建依赖）

```bash
scene-factory studio --port 8788
```

浏览器打开 `http://127.0.0.1:8788`，提供三视图舞台（3D 视口 + MuJoCo 离屏渲染 + 实时可编辑 XML）、零渲染 SE(3)/OBB 3D 穿模诊断叠层、多轮 Agent Trace 时间线、原地 Random 变体库与 Cordis 插件热开关。

### 4. 纯 XML 空间坐标与 3D OBB 碰撞检测 / S-Space 基准

```bash
# 零渲染直接解析任意 MuJoCo XML 的嵌套 SE(3) 坐标系与 3D OBB 碰撞
scene-factory spatial-check outputs/table_mug_scene/scene.xml --crosscheck-mujoco

# 生成 3 个难度的 XML-Spatial-Collision 基准集（兼容 mirros-lab/s-space）
scene-factory benchmark-xml --output outputs/xml_spatial_bench --per-level 30
```

### 5. 作为标准 MCP 插件挂载给外部 Agent（如 Cordis / DeepSeek Harness / Claude / Cursor）

```bash
scene-factory mcp
```

暴露 `search_assets`、`compose_assets_xml`、`verify_mujoco_xml`、`randomize_mujoco_xml` 四个标准 JSON-RPC 2.0 MCP 工具，零外部依赖。

---

## 开发测试

```bash
python -m ruff check scene_factory tests
python -B -m compileall -q scene_factory tests
python -B -m pytest -p no:cacheprovider -q
```

## 许可证

代码采用 MIT 许可证；随仓提供的第三方资产署名见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
