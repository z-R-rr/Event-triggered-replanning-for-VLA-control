# Replan router canonical pipeline v1

> Superseded. This split-and-shortlist design must not be used for input-feature
> selection. Use
> [Replan router input-feature selection v2](REPLAN_ROUTER_FEATURE_SELECTION_V2.md).

唯一入口：

```bash
python3 robotwin/script/run_replan_router_pipeline.py --stage validate
```

固定配置是
[`configs/replan_router_pipeline_v1.json`](../configs/replan_router_pipeline_v1.json)。
不带 `--execute` 时，耗时或写文件的 stage 只打印命令；带
`--execute` 才实际运行。

## 1. 验证冻结输入

```bash
python3 robotwin/script/run_replan_router_pipeline.py --stage validate
```

必须同时通过：

- dataset fingerprint：`36331e...f400`，348 samples / 89 scenes；
- scene-grouped split hash：`4a9a...8de4`；
- train/val/test：242/54/52 samples，62/13/14 scenes；
- 所有 module 使用同一 dataset、split 和 label；
- shortlist checkpoint 的 SHA-256 与配置一致。

任一检查失败都停止，不进入训练或 online。

## 2. 训练全部 module

先预览 10 条确定性训练命令：

```bash
python3 robotwin/script/run_replan_router_pipeline.py --stage offline
```

确认后运行：

```bash
python3 robotwin/script/run_replan_router_pipeline.py \
  --stage offline \
  --execute
```

训练固定项：frozen pi0.5 vision encoder、两头 MLP、AdamW、batch 32、
最多 200 epochs、patience 30、train/split seed 0。输出写入
`temp/outputs/replan_router_pipeline_v1/offline/`，不覆盖旧实验。
只有 10 个 evaluation 和 checkpoint 全部存在后，后续 shortlist/online
才会整体切换到该目录；部分完成时继续读取 frozen reference，禁止混用。

仅验证数据和 split 时使用：

```bash
python3 robotwin/script/run_replan_router_pipeline.py \
  --stage dataset \
  --execute
```

## 3. 计算 offline 指标并冻结 shortlist

```bash
python3 robotwin/script/run_replan_router_pipeline.py \
  --stage shortlist \
  --execute
```

选择规则：

- 只有完整 52-sample test cohort 可参加横向排序；
- 参数量必须不超过 1M；
- V4/V6/V7 因 temporal history 缺失后只剩 13 samples，仅作 diagnostic；
- V0 固定为 baseline；
- 其余按 F1 降序、FTR 升序、AUC 降序、参数量升序取前 2 个。

v1 预期 shortlist 是 `V0, E1, V1`。结果写入
`temp/outputs/replan_router_pipeline_v1/configs/online_shortlist.json`。
shortlist 只读取 offline 指标，不读取任何 online result。

## 4. 冻结 unseen-20 并检查 online plan

选择新的 expert-valid scenes：

```bash
python3 robotwin/script/run_replan_router_pipeline.py \
  --stage select-scenes \
  --execute
```

然后生成 80-episode plan，但不启动 server：

```bash
python3 robotwin/script/run_replan_router_pipeline.py \
  --stage online-prepare \
  --execute
```

固定协议：control + 3 routers、每组 20 scenes、H=50、r0=25、
query interval=5、max replans=1、lambda=0.05、GPU 0–3、port 8400–8403。
必须在启动前检查 `experiment_plan.json` 的 scene manifest hash、
checkpoint hash、arm、GPU 和 port。

## 5. Online eval 与最终判定

```bash
python3 robotwin/script/run_replan_router_pipeline.py \
  --stage online-run \
  --execute
```

eval 正常启动后无需轮询；各 arm 会从缺失的 `metrics.json` 继续。全部结束后：

```bash
python3 robotwin/script/run_replan_router_pipeline.py \
  --stage summarize \
  --execute
```

候选只有在 20/20 pairs 有效、所有 pair audit checks 通过、
`router_successes > control_successes` 且 `rescues > harms` 时才晋级。
平局不晋级。failure-only threshold sweep 只能作为诊断，不能估计
false-trigger harm，也不能修改 v1 的 lambda。

任何 dataset、split、label、feature 定义、训练协议、排序规则、
threshold 或 online scene set 的改变，都必须复制配置为新的
`pipeline_id`，并重新生成 unseen scene manifest。
