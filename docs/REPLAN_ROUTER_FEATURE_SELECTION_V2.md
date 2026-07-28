# Replan router input-feature selection v2

唯一入口：

```bash
python3 robotwin/script/run_replan_router_feature_selection_pipeline.py \
  --stage validate
```

固定配置：
[`configs/replan_router_feature_selection_v2.json`](../configs/replan_router_feature_selection_v2.json)。
不带 `--execute` 时只验证或打印命令，不启动训练和 online eval。

## 1. 固定 Offline 数据池

所有主候选使用同一组 348 counterfactual samples：

|Transition|Samples|
|---|---:|
|rescue|56|
|harm|16|
|both_success|276|

共 89 scenes，both-failure 已排除，dataset fingerprint 为
`36331e...f400`。

主候选固定为 `V0, V1, V2, V3, V5, E1, E2`。V4/V6/V7 会因缺少
`t-k` observation 跳过 sample，无法满足相同 348-sample cohort，因此
只允许单独做 diagnostic，不能进入 Result 1/2 主表。

## 2. Result 1：scene-grouped OOF

先验证已经冻结的五折：

```bash
python3 robotwin/script/run_replan_router_feature_selection_pipeline.py \
  --stage offline-prepare
```

实际训练：

```bash
python3 robotwin/script/run_replan_router_feature_selection_pipeline.py \
  --stage offline-train \
  --execute
```

协议：

- 5-fold deterministic scene-grouped OOF，按 scene transition 分层；
- 同一 scene 不会同时出现在一个 fold 的 train/val/test；
- 每个 sample 恰好产生一次 out-of-fold prediction；
- Result 1 在合并后的 348 个 OOF predictions 上计算；
- 每个 online checkpoint 再用全部 348 samples 重训。

最终重训 epoch 数取五个 fold 最佳 epoch 的中位数四舍五入。最终
checkpoint 的 normalization 也只基于这 348 samples。所有 feature 使用
同一 folds、optimizer、seed、batch size、early stopping 和 threshold。

## 3. 固定 balanced Online cohort

cohort 已由已有的、训练未见的 confirmed-control eval 构造：

```bash
python3 robotwin/script/run_replan_router_feature_selection_pipeline.py \
  --stage cohort
```

组成：

- 20 个 control 已确认失败 scenes；
- 20 个 control 已确认成功 scenes；
- 两个 stratum 内均按 scene seed 排序取前 20；
- 40 scenes 与 Offline 89 scenes 完全不相交；
- cohort identity：`04e179...d6cab`。

固定 seed、prompt、control outcome、source metrics hash 和 source summary
hash 均记录在
`temp/outputs/replan_router_feature_selection_v2/online/inputs/`。

## 4. Result 2：所有 feature 共用 Online-40

生成两批计划但不启动 eval：

```bash
python3 robotwin/script/run_replan_router_feature_selection_pipeline.py \
  --stage online-prepare \
  --execute
```

两批设计：

|Batch|GPU arms|新 episodes|
|---|---|---:|
|batch_0|control + V0 + V1 + V2|160|
|batch_1|V3 + V5 + E1 + E2|160|

batch_1 直接复用 batch_0 的 40 个 immutable control cases。总共执行
`40 control + 7 × 40 router = 320` 个新 episodes。

每个 feature 固定 H=50、r0=25、query interval=5、max_replan=1、
lambda=0.05、absolute-r0 cadence、policy seed=0。结果按 frozen control
stratum 分开报告：

- control-failure：router success rate 和 rescues；
- control-success：success retention 和 harms；
- overall：success rate、trigger rate、40/40 pair validity；
- control rerun 必须与选择时的 confirmed outcome 一致。

## 5. 运行与汇总

先运行 batch 0：

```bash
python3 robotwin/script/run_replan_router_feature_selection_pipeline.py \
  --stage online-run \
  --batch batch_0 \
  --execute
```

batch 0 完整结束后再运行 batch 1：

```bash
python3 robotwin/script/run_replan_router_feature_selection_pipeline.py \
  --stage online-run \
  --batch batch_1 \
  --execute
```

最后生成 Result 1 + Result 2 联表：

```bash
python3 robotwin/script/run_replan_router_feature_selection_pipeline.py \
  --stage summarize \
  --execute
```

在 Result 2 完成前不基于 Offline 指标淘汰 feature，也不在 Online-40
上重新扫 threshold。最终 feature 选择依据两张并列表人工确认，不使用
未预注册的加权分数。
