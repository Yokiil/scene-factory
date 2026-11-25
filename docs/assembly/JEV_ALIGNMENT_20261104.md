# JEV 与 SceneFactory 的结合方式与验收边界

核对日期：2026-11-04。这里的 JEV 指用户给出的 Benchmark Heaven Jev-class 模型榜及
TypeSafe 的 Jev/System One 决策协议。本轮没有得到 TypeSafe API key，没有调用付费模型，
没有下载或训练榜单模型。接口测试替身与真实模型推理在证据中明确区分。

用户提供的[模型榜](https://benchmarkheaven.com/jev-models)可以帮助找到决策模型、模型卡与案例。
本项目参考的是状态表达、有界选项、概率校准和技能筛选方法；榜单排名与案例推理命令本身
不构成训练方案。当前已交付接口和离线数据出口，真实模型对照与模型训练仍待执行。

## 与老师要求的关系

结合方向切合，但 JEV 应放在物理沙盒的上层：输入当前测量状态、任务约束、候选技能和失败历史，
输出有限选项中的下一技能、风险评分或恢复判断；下层控制器继续产生受限关节动作并执行真实接触，
物理引擎和轨迹判据判定结果。不能由模型的自评分替代装配深度、接触力、释放稳定等测量。

Benchmark Heaven 的方法页把 JevBench 定义为状态与有界 rubric 到结构化决策的评测，衡量
准确率、校准、延迟和成本。它并不是机器人技能、接触求解或物理保真度榜。其自托管延迟与 API
网络端到端延迟还采用不同口径，不能拿榜单数字作为本项目机器人控制频率。

来源：[JevBench 方法页](https://benchmarkheaven.com/jev-models/methodology)、
[官方快速开始](https://docs.typesafe.ai/introduction/quickstart.md)。网页会更新；本说明不引用实时名次
作为选型结论，也不将方法页的历史 reviewed 日期当成本次访问日期。

## 可以学哪些公开案例

| 公开案例 | 本次在原始资料中核实到的内容 | 对本项目的借鉴和限制 |
| --- | --- | --- |
| TypeSafe Skill suggestion | 对候选技能先筛选，再读取少数候选详细说明并允许全部拒绝；有 baseline/建议/oracle 对照 | 可迁移成“观察/抓取/重新对齐/撤回/中止”的技能选择评测。其原实验是软件 agent 技能加载，不是机器人装配成功率；不复制其收益数字 |
| Cloudflare Clef | 固定模型卡说明基于 Qwen3.8-27B 后训练，联合 schema head 对每个问题的选项给出 logits；包含票据和图像输入示例 | 可学测量状态与候选选项的联合表达、多问题批处理、输出概率。模型卡没有给出足以复现全部后训练的完整数据/超参数，不能称已复刻训练方法 |
| H2O-Lightning-4B | 固定模型卡说明 Qwen3.5-4B、单次前向决策及 System One shim，公开 ticket 优先级例子和复现实验命令；自述未用 JevBench 数据训练 | 可学严格协议校验、真实模型版本锁定、近似平局和硬件数值差异记录。该固定旧卡主要是推理/评测说明；后续训练概况见下方单独固定的新版本。本机未运行，也未测量量化配置及与 Isaac 共享显存的可行性 |
| LiquidAI d1-3B | 固定模型卡给出 LFM2.5-VL-3B 后训练决策模型、文本/图像、多问题前向接口 | 可学多模态测量摘要和边缘部署候选。其公开性能数字来自厂商指定设备/输入，本项目未复测；仅参考卡片不能推导完整训练配方 |

固定来源：

- [TypeSafe 技能选择案例](https://docs.typesafe.ai/cookbooks/skill_suggestion.md)。这是应用案例，不是训练教程。
- [Clef 模型卡，revision 2f3de3dd85f379784083b0814d997ab627200f0c](https://huggingface.co/Cloudflare/clef/blob/2f3de3dd85f379784083b0814d997ab627200f0c/README.md)
- [H2O-Lightning-4B 模型卡，revision 193ad740925b176a3b70a5a13a7cff2f2fadd01e](https://huggingface.co/h2oai/h2o-lightning-4b/blob/193ad740925b176a3b70a5a13a7cff2f2fadd01e/README.md)
- [d1-3B 模型卡，revision 051bcc464b01b9f92942b364d9586b0ef5912432](https://huggingface.co/LiquidAI/d1-3B/blob/051bcc464b01b9f92942b364d9586b0ef5912432/README.md)

要做物理 AI 训练，应从本沙盒采集 state→候选技能→实际动作轨迹→物理结果，用真实引擎与冻结的
任务判据标注标签。先用规则/参考控制器建立基线，再比较模型在相同 seed、资产族隔离划分下的
成功率、超力率、拒答率、校准和延迟。不得以复现公开榜单或训练其 public split 替代本任务测试。

## 可以进一步复现的训练方法

2026-11-04 补充核对了两份作者公开的方法，区分可学的方法与本项目已经执行的实验：

- H2O 的 [Lightning 模型卡，revision 659fc412de8486dbc2477b0d1cb4121e97e72471](https://huggingface.co/h2oai/h2o-lightning-4b/blob/659fc412de8486dbc2477b0d1cb4121e97e72471/README.md) 说明按阶段混合通用、
  文档、数值、游戏和安全等决策样本。可借鉴按选项进行监督，以及分开记录数据类型与题型。
  此新版本披露训练阶段、样本组成和题型比例，与上表旧 revision 分开理解；这些概况仍不足以
  完整重建全部训练数据、样本生成过程和超参数，不据此声称已复刻作者训练。
- [Releasing the Inner Jev，v1](https://arxiv.org/html/2610.17205v1) 给出把通用模型适配成有界决策模型的
  LoRA 路径：从语言模型字母 token 的输出行初始化决策头，对可选项计算概率，再训练适配器与
  决策头修正。作者提供 [Jeff 代码，commit 47ac025a1d8a011bdb1fb1cbbbf20607884d35ee](https://github.com/j-geuter/Jeff/tree/47ac025a1d8a011bdb1fb1cbbbf20607884d35ee)。可参考其方法设计
  技能选择实验，再比较基础模型与适配模型；公开论文结果不能替代本项目机器人实验。

这里新增的是原始资料核对与训练方法参考。本轮未安装训练栈、下载模型权重或执行上述训练。
本地出口目前只有一个资产族/任务结构分组，因此先补独立分组及失败/拒答标签，再进行有独立
验证集的训练和模型比较；不能把现有 512 行行为模仿标签包装成已完成的 JEV 训练结果。

迁移到精细装配时，下一轮实验应先冻结可执行技能、测量字段和标签来源，再收集独立资产族/
任务结构的成功、恢复与失败轨迹。`observed_behavior` 只证明执行过某技能；要学习“应选哪项”，
需要额外审核参考控制器决策，或在相同锁定环境和初始条件下比较候选技能的实测结果。
当前 Isaac 不支持任意状态快照，不能把尚不存在的精确分叉恢复当作标签生成前提。
可先以选项分类训练建立基线，在独立验证集校准概率，再用固定测试场景比较规则控制器、基础
模型和适配模型的任务成功率、超力率、拒答率与决策延迟。以上是待执行实验设计。

## 本轮实现

`scene_factory/agent_adapter.py` 提供：

1. `RuntimeAgentAdapter`：只经公开 HTTP create/reset/step/close 执行外部 policy，记录每次动作、
   状态版本、策略延迟和真实任务结果，异常也释放 session。
2. `JevHttpTransport`：按官方 `POST /v1/systemone` 发送 state/questions/model，密钥从环境变量
   `TYPESAFE_API_KEY` 读取；不记录密钥或供应商错误体。支持配置兼容 endpoint，模型实际版本取响应值。
3. `JevSkillSelector`：只允许注册技能；校验 Choice/Score/Noul 类型、候选集、有限概率与和、
   风险分数、置信差距；证据缺失、概率接近、风险过高或恢复证据不足时拒答。
4. `JevDecisionPolicy`：从真实观测构造决策状态，物理超力直接停止；模型仅选择代码预注册的固定动作，
   不能生成执行代码、瞬移零件或任意力矩。模型输出不具有任务成功裁决权。

示例执行：

```sh
python tools/validate_agent_randomization.py --output outputs/agent_contract_smoke --candidates 12
```

默认使用明确标名的 `contract_test_double_not_jev_inference`，但 HTTP/MuJoCo 是真实运行。
凭证和网络可用时另建输出目录并加 `--live-jev --model <实际可用版本>`。
不承诺 `jev-latest` 永久指向相同模型；本项目会记录响应中的 `model`。

本地证据：`outputs/completion_20261104/agent_randomization_final/report.json`。
12 个带四层随机化的候选经 HTTP 运行一个外力抬升代理任务，12 个达到该任务的抬升门槛。
这是接口与采样执行证据，不能算机器人抓取、维修或精细插接成功，也不能证明真实 JEV 策略有效。
本轮模型真实调用状态为 **未验收**。

## 四层随机化的实际支持范围

`reset.options.randomization` 支持独立 RNG 的 layout、physics、initial_state、observation：

- layout：已有动态叶子实例的 XY/yaw 覆盖或抖动；尚不支持 episode 中替换资产族/拓扑。
- physics：符合约束的 box/sphere/cylinder 代理实例质量、惯量、摩擦和关联尺度，MuJoCo 实际编译使用。
- initial_state：具有实际初始位姿实现的 XY/yaw 错位；不虚称电气故障、柔性卡扣或关节卡滞已实现。
- observation：独立高斯位置测量噪声和可选 RGB 噪声。原始物理真值保留在原字段，噪声测量独立存放；
  任务成功从原始物理状态判断。同 seed、同状态索引得到同噪声；重试和同后端状态恢复保持一致。

不支持的变量明确拒绝；Isaac 尚不支持上述 episode 动态重建随机化，能力检查仍拒绝。
各层实际采样、配置摘要、seed 和难度标签写入 reset 证据。难度标签是配置分层，不是已证明的
策略难度排序。`assign_family_splits` 先把共享资产族或任务结构的记录连成分量，再分配数据集；
当前演示只有一个共享资产/任务分量，因此明确报告无独立验证/测试集，不通过换 seed 伪造划分。

## 训练数据出口

`decision_dataset.py` 和 `tools/export_decision_dataset.py` 已提供离线严格导出与复核。
现有 12 个真实 HTTP/MuJoCo episode 导出 **512 行** `state/rubric/observed_behavior` 样本，
包含任务结果、失败标记、源轨迹/动作摘要、模型来源和资产族/任务结构分组。
证据位于 `outputs/completion_20261104/decision_dataset_final/manifest.json`；调用说明见
`docs/DECISION_DATASET.md`。标签是执行过的技能，未证明最优；本次没有训练模型。
数据只有一个共享分组，不能从此数据报告独立测试集泛化结果。
