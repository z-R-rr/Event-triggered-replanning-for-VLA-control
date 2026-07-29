# `pick_dual_bottles`：Z2-Cross-v3.1 跨任务复现 handoff

## 0. 范围与冻结定义

目标是验证当前标准 router module `Z2-cross-v3.1` 能否在一个新的任务
`pick_dual_bottles` 上，以**任务内训练、任务内未见场景评测**复现正向 online
净收益。

本轮不是 joint multi-task training：不把 `move_playingcard_away` 的 pair、特征
或 Online-40 scene 混入本任务。跨任务结论只比较两个独立任务在同一 module
与协议下的表现。

固定：

```text
task:              pick_dual_bottles / demo_clean
H:                 50
r0:                25
query interval:     5 completed actions
max replans:        1
lambda:             0.05
cadence:            absolute r0
policy seed:        0 + deterministic torch
split:              5-fold scene-grouped, split seed=0
```

v3.1 固定监督：

```text
beneficial = rescue
neutral    = both-success + both-failure
harmful    = harm
window_score = P(beneficial) - P(harmful)
batch = rescue 10 + both-success 5 + both-failure 5 + harm 10
optimizer = AdamW(lr=1e-3, weight_decay=1e-4)
batch size = 32; max epochs = 200; patience = 30
```

`pi0.5` vision/action-expert 均冻结；router 输入保持 Z2-Cross：当前视觉 token、
old remaining action-expert hidden tokens，以及 CrossAttn alignment。

同一 scene 中的 rescue / both-failure（BF）相邻节点关系必须保留在数据 manifest
中，但本轮**不**加入 pairwise ranking loss，也不改变 v3.1 的三分类
cross-entropy 目标。它们仅作为后续 within-scene ranking diagnostic 的可追溯证据。

## 1. Episode 预算与完成门槛

下表中的“policy episode”表示一次完整 RoboTwin policy rollout；“pair”是一个
共享 control 与一次 forced replan 构成的 counterfactual 训练样本。二者不得混称。

| 阶段 | 目的 | 新增 policy episodes | 固定产物/门槛 |
|---|---|---:|---|
| A. checker 静态审计 | 审核双瓶任务的 success 语义 | 0 | `check_success` 同时检查两瓶的目标 xy 与 `z>0.89`；记录潜在误判风险 |
| B. training r25 shared controls | 100 个训练 scene 的统一 control trunk | 100 | 每个 scene 一条 deterministic r25 trace；保存所有 11 个粗节点 observation |
| C. paired reproducibility smoke | 验证 control/forced 的 prefix、observation、cadence 合同 | 6 | 3 control + 3 forced；`passed=true` 后才可启动大规模 forced episodes |
| D. coarse forced grid | 任务无关的第一轮 counterfactual labels | 1,100 | 100 scene × 11 nodes；control 复用 B，不额外执行 |
| E. boundary/random refinement | 增加 rescue/BF 边界与随机 harm 反例 | ≤288 | 只新增 forced episodes；规则与预算在下方冻结 |
| F. feature extraction + OOF/final training | 离线训练 v3.1 | 0 | 5 OOF models + 1 final checkpoint；只读取 B/D/E 的保存数据，不推进环境 |
| G. disjoint r25 online screening | 构造平衡 Online-40 control cohort | 100 | 100 个不在 offline train scene 中的 r25 controls；筛出 20 failure + 20 success |
| H. real-server online smoke | 验证 router server/PID/checkpoint 与触发链路 | 2 | 1 control + 1 router pair，端口与 checkpoint identity 均通过 |
| I. frozen Online-40 | Result 2 主评测 | 80 | 40 controls + 40 v3.1 router episodes；若 G 的 selected 40 controls 可按完全相同协议复用，则新增仅 40 router episodes |
| J. semantic outcome review | 排除 checker-only success | 0 | 复核 I 的所有 router-success trace；输出 checker 与语义裁决的差异 |

第一轮实际为 `100 shared controls + (100 × 11 = 1,100) forced = 1,200`
policy episodes；`1,200` 不是 forced episode 数。第二轮最多再加 288 条
forced episode。

**保守总预算：1,676 policy episodes。** 其中 I 按重新执行 40 个 controls
计算；若 G 的 40 个 selected controls 通过 hash/prompt/trace identity audit 并被
合法复用，则总计为 **1,636**。A、F、J 不运行 policy episode。

可选诊断（不属于训练或主评测）：对 20 个 Online failure 做每 scene 5 个手动候选
node oracle，需 `20 shared controls + 100 forced = 120` policy episodes；它只给出
timing 上界，不能用于训练、调阈值或选择 Online-40 scene。

## 2. 两轮训练数据收集与 quota gate

### 2.1 第一轮：所有 scene 的粗网格

100 个 training scene 各运行一条 r25 shared control，然后对每条 control 的
同一 prefix 强制在以下 11 个节点各 replan 一次：

```text
t = 10, 15, 20, 30, 35, 40, 45, 55, 60, 65, 70
```

每个 forced case 与该 scene 的同一条 control 配对，标签严格由 paired outcome
给出：

| Control | Forced at t | Transition |
|---|---|---|
| failure | success | rescue |
| success | failure | harm |
| success | success | both-success |
| failure | failure | both-failure |

若 control 在节点 `t` 前已经 terminal、没有记录到 node observation 或未通过
paired validity gate，该节点是 unavailable，不执行或不计入训练 pair；因此
`1,100` 是 forced rollout 上限，训练样本数以 valid pairs 为准。

### 2.2 第二轮：边界加密与随机 harm 反例

第二轮总预算最多 288 条 forced episode，不对所有 scene 做 dense scan。

- failure scene：对相邻粗节点出现 `rescue ↔ both-failure` 翻转的区间加密。
  例如 `t=15 rescue`、`t=20 both-failure` 时，已有节点 `10,15,20` 不重跑；
  新增合法节点为 `11,12,13,14,16,17,18,19`，共 8 条。自然边界 `25,50,75,...`
  一律排除。
- success scene：每个 eligible success scene 用固定随机种子额外抽两个节点：
  一个来自早期 `11–34`，一个来自晚期 `36–69`；排除 natural boundaries 与已经
  跑过的粗节点。抽样不读取 router score，不按模型怀疑程度挑选节点。
- 所有候选在执行前写入 immutable `round2_schedule.json`；failure boundary
  jobs 按 scene round-robin 排序，success random jobs 的 seed、候选池和抽中节点
  一并保存。达到 288 条即停止，未执行项保留为 pending，而非重新排序。

success 的随机节点用于保留 harm 的自然发生率证据；boundary-densified failure
节点用于丰富 rescue/BF timing 边界。两种来源在 manifest 中必须显式区分，不能把
enriched 训练集的类别比例解释为任务的自然发生率。

### 2.3 最低训练数据配额

开始 F 前，valid pair 数据集必须至少满足：

| Transition | counterfactual pairs | v3.1 中的类别 |
|---|---:|---|
| rescue | ≥120，且覆盖 ≥40 个 control-failure scenes | beneficial |
| harm | ≥60，且覆盖 ≥30 个 control-success scenes | harmful |
| both-success | ≥200 | neutral 来源 |
| both-failure | ≥200 | neutral 来源 |
| 总计 | **≥580** | scene-grouped cohort |

v3.1 的 batch 组成仍固定为 `rescue 10 + both-success 5 + both-failure 5 + harm 10`；
训练使用所有 valid pairs，而不是为了回到旧任务的 476 pairs 而下采样。当前训练器
的旧任务计数 assertion 必须在开始 F 前改为读取本任务冻结 manifest 的这些最低
配额与实际计数。

固定 100 scene 粗网格与 288 refinement 只能给出最大 `1,388` 个 forced pair
机会，不能数学上保证上述 outcome quota。若 E 完成仍不足任一 quota，停止训练：

1. 新建（不可覆盖）扩展 round output root；
2. 继续使用相同、预注册的 coarse / boundary / random 规则，并保持 Online pool
   scene-disjoint；
3. 只在满足所有 quota 后开始 F；
4. 不降低类别门槛、不复制 pair 伪造样本量、不把任务间 pair 混入。

## 3. 数据划分和训练步骤

1. B/D/E 的 100-scene pool 只服务于本任务的 training-pair candidate discovery。
2. E 完成后冻结所有 valid pair 的 sample ID、scene group、round/source、prompt
   plaintext/SHA、old chunk SHA、node observation fingerprint。
3. 以 scene 为 group 做 5-fold OOF；每个 pair 恰好一次 test prediction。
4. 每折由训练 scene 的 15% 组成 inner validation；final checkpoint 用全部达到
   quota 的有效 pairs，训练 epoch 为五折 best epoch 的 rounded median。
5. F 的固定输出包括：feature archive/manifest、fold identity、normalization、
   checkpoint SHA、OOF predictions、evaluation JSON。

同一 scene 的 rescue/BF 邻接关系另外写入 `within_scene_relations.jsonl`，至少包含
`scene_seed`、`rescue_node`、`both_failure_node`、round/source 和两条 pair 的
fingerprint。该文件不进入 v3.1 loss，也不改变 batch sampling；只用于报告
`score(rescue) > score(BF)` 的 post-hoc within-scene ranking 指标。

Offline Result 1 必报：params、ROC-AUC、accuracy、F1、strong rescue recall、
false trigger、precision，以及四个 transition 的 trigger count/rate。

## 4. Online-40 设计

G 的 100 个 r25 screening scenes 必须与 B/D/E 的 training scene groups 不重叠。
按 control outcome 独立按 seed 排序，各取前 20 个，冻结：

```text
20 confirmed control failures
20 confirmed control successes
40 resolved prompts + SHA-256
40 control traces / outcome audit
```

H 中每一个 router pair 必须验证：

- scene seed、prompt plaintext 和 prompt SHA 相同；
- 首次 trigger 前 action targets 与 observations 等同 control；
- query 仅在 interval-5 且不与 natural r0 boundary 重叠的合法节点；
- 最多一次 trigger，replacement 用 trigger observation，absolute cadence 正确；
- listener PID、router checkpoint path 与 SHA-256 匹配。

Result 2 必报：`valid_pairs`、control outcome match、router success、rescue、
harm、both-success、both-failure、failure/success strata success rate、trigger rate。

## 5. `pick_dual_bottles` success 语义门槛

当前环境 checker 要求两瓶分别接近左右 target xy，且两个 functional point 均
`z>0.89`；这与任务描述“无需松爪”一致。静态审计必须将此写入 manifest。

在 I 完成后，不修改运行中的 checker；对每个 checker-success 保存 head/left/right
camera 末帧及最后状态，人工确认：

1. 两瓶都存在、未掉落或离开工作区；
2. 左瓶在左 target、右瓶在右 target；
3. 成功不是单瓶或错误交叉放置造成的偶然 checker 命中。

最终汇总同时给出 `checker_success` 和 `semantic_success`。在语义复核前，任何
Online-40 成功率都标为 checker-only。

## 6. 输出根与目录

所有新产物使用独立根，禁止覆盖现有 `move_playingcard_away` 结果：

```text
temp/outputs/replan_router_z2_cross_v31_pick_dual_bottles_v1/
├── discovery/
├── paired_smoke/
├── counterfactual_train/
├── features/
├── offline/
│   ├── configs/
│   ├── folds/
│   ├── checkpoints/
│   ├── evaluations/
│   └── logs/
├── online/
│   ├── inputs/
│   ├── smoke/
│   ├── control/
│   ├── Z2-cross-v3.1/
│   └── semantic_review/
└── logs/
```

### 6.1 固定四组并发拓扑

第一轮 shared-control 和 coarse forced grid 使用四个独立的 server/eval
队列。一个 scene 的 shared control 以及该 scene 的全部 11 个 forced nodes 必须
始终落在同一 group/shard；不得跨 group 重新生成 control，也不得把同一 scene 的
nodes 分给不同 queue。

| Group | Server GPU | Eval/client GPU | Port | 训练 scene |
|---|---:|---:|---:|---:|
| 0 | 0 | 1 | 8600 | shard 00，25 |
| 1 | 0 | 1 | 8601 | shard 01，25 |
| 2 | 2 | 3 | 8610 | shard 02，25 |
| 3 | 2 | 3 | 8611 | shard 03，25 |

每个 shard 的粗网格工作量固定为 `25 controls + 25 × 11 = 300` policy
episodes；四组总计仍为 1,200，不因并行而改变数据或 split。每个 group 只启动
一个 OpenPI server 和一个串行 evaluator；GPU1/3 不额外 colocate server。端口、
server PID、checkpoint path/SHA 与 shard manifest 都写入运行日志和 topology
manifest。开始大规模 coarse forced 前，四个 group 均需通过 paired smoke 的
server/client 连通性与可复现性门槛。

## 7. 执行顺序与 stop gates

```text
A checker audit
  -> B r25 shared controls (100)
  -> C paired smoke (6)
  -> D coarse forced grid (1,100)
  -> E boundary/random refinement (<=288)
  -> exact quota + validity + scene-disjoint audit
  -> F offline OOF/final training (0)
  -> G disjoint r25 screening (100)
  -> freeze 20-failure + 20-success cohort
  -> H online smoke (2)
  -> I Online-40 (80, or reuse-audited-control 40)
  -> J semantic success review (0)
  -> cross-task comparison with move_playingcard_away v3.1
```

任何以下情况均 fail closed：不足四类 outcome quota、训练/online scene overlap、
prompt SHA 不一致、invalid pair、server checkpoint/PID 不匹配、或 semantic review
发现 checker-only success 未单列。

## 8. 当前状态

```text
status: planned; no pick_dual_bottles control grid, pair dataset, checkpoint, or online result exists yet.
next action: A + B（静态 checker audit 后创建 100 条独立 r25 shared controls）。
```
