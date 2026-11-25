# 测量轨迹到决策训练数据

`scene_factory.decision_dataset` 把已保存的 `agent_rollout.v1` 轨迹导出为 `state + rubric + label` JSONL。它不执行训练，也不调用 JEV。每一行的输入是动作执行前的状态；标签是记录中实际选中且已执行的固定技能。`label.kind=observed_behavior`、`optimality=not_established` 不允许改写成“最优动作”。物理任务的最终成功、超时或失败单独保存。

导出要求 MuJoCo 或 Isaac 的会话观测、连续的 episode/state/sequence/time、与注册技能一致的动作、对应决策记录和明确的策略来源。dry-run、决策与执行不一致、跨 episode 混合、非有限数、未执行的动作标签都会拒绝。失败轨迹保留 `failure_flags`；失败后产生但未成功执行的决策计入 `ignored_unexecuted_decisions`，不会成为训练行。当前导出只支持 `selected` 技能记录，拒答与安全回退需定义独立的人工审核标签后再扩展。

每个 source 必须显式声明资产族和任务结构。数据在 episode 层按“共享资产族或任务结构”的连通分组划分，再展开到动作行。同族或同结构不能跨训练/验证/测试集。小型数据只有一个分组时，验证/测试可以为空；不为凑比例拆开分组。

配置文件示例：

```json
{
  "split_seed": 0,
  "sources": [
    {
      "path": "episode-0000.json",
      "asset_families": ["mug_proxy"],
      "task_structure": "external_force_lift",
      "policy_source": "contract_test_double"
    }
  ],
  "skills": [
    {
      "skill_id": "bounded_lift",
      "description": "Apply the registered bounded upward force to the proxy",
      "action": {"mode": "object_forces", "payload": {"object_forces": {"module": [0, 0, 6]}}}
    },
    {
      "skill_id": "wait",
      "description": "Observe passive physical evolution",
      "action": {"mode": "noop"}
    }
  ]
}
```

```powershell
.venv\Scripts\python.exe tools\export_decision_dataset.py --config export_config.json --output outputs\decision_dataset
.venv\Scripts\python.exe tools\export_decision_dataset.py --verify --output outputs\decision_dataset
```

输出包含 `samples.jsonl`、`family_splits.json` 和 `manifest.json`。清单锁定源轨迹和 JSONL 的 SHA-256，记录分组数量、失败类别、未执行决策、空划分和训练状态。`validate_decision_sample` 进行严格字段及语义验证；`verify_decision_dataset` 检查文件摘要、行唯一性和跨行分组泄漏。公开 JSON Schema 位于 `schemas/decision_training_sample.schema.json`。源轨迹摘要证明记录未变，不能单独证明记录来自可信的仿真采集器；使用者仍需信任采集流程并保存原始运行证据。

`policy_source` 支持 `contract_test_double`、`typesafe_endpoint`、`self_hosted_jev`、`other_external_policy`。已知测试替身必须标注 `test_double_no_live_inference`。外部模型来源仅标注 `reported_in_rollout_not_independently_verified`，导出器不会把字符串来源当成独立的在线推理验收。

2026-11-04 的交付样例来自本机真实 HTTP/MuJoCo 四层随机化外力抬升轨迹，决策由明确标注的测试替身产生。它支持检查数据管线和模仿标签，不能证明 JEV 模型质量、机械臂装配能力或泛化。尚未训练任何模型。
