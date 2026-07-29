# pi0.5 RoboTwin deterministic one-replan pipeline

This is the execution index for the validated pipeline. The algorithmic
rationale and known failure modes remain in
`script/PI05_REPLAN_CAUSAL_PIPELINE_HANDOFF.md`.

## 1. Frozen protocol

| Item | Value |
|---|---|
| Checkpoint | [`motus-robotics/pi0.5_robotwin2`](https://huggingface.co/motus-robotics/pi0.5_robotwin2), installed at `/home/ubuntu/Model/pi0.5_robotwin2` |
| OpenPI config | `pi05_robotwin2_multitask_pytorch` |
| Task config | `demo_clean` |
| Predicted horizon | `H=50` |
| Discovery execution lengths | `r=10,15,25,30,35,40` |
| Fixed reference | `r0=25` |
| Scenes | ordered fixed manifest of 100 |
| Torch mode | deterministic from discovery onward |
| Candidate time tolerance | 2 actions |
| Candidate window radius | 2 actions |
| State proxy | preceding five executed 12-D arm-joint targets |
| Maximum state distance | 0.35 |
| Window cap | 3 families per scene |
| Node cap | 10 nodes per scene |
| Smoke | control 3 repeats + forced 3 repeats |

Node `t` means: execute actions `1..t`, discard the old chunk tail, and infer
again immediately before one-based action `t+1`.

## 2. Code ownership

Runtime and reproducibility:

| File | Responsibility |
|---|---|
| `policy/pi05_remote.py` | Trace capture, prompt fingerprint, occurrence token, node observation, and forced replan before action `t+1` |
| `script/eval_policy_wandb.py` | Fixed seed manifest, resolved prompt manifest, and one-episode RoboTwin execution |
| `script/grid_search_pi05_hr.py` | Six-r launcher, deterministic OpenPI server, resume logic, and grid result collection |
| `/home/ubuntu/Workspace/openpi/scripts/serve_robotwin_policy.py` | Deterministic Torch inference server |

Analysis and evaluation:

| Stage | Script | Main output |
|---|---|---|
| Prompt audit | `recover_pi05_prompt_manifest.py` | `prompt_recovery_report.json` |
| Contrast selection | `make_pi05_outcome_subset_manifest.py` | `seed_manifest_r25_fail_other_success.json` |
| Window/state mapping | `build_pi05_replan_window_controls.py` | `replan_window_controls_r25_fail_other_success.json` |
| Concrete nodes | `extract_pi05_replan_nodes.py` | `replan_nodes_r25_fail_other_success.json` and CSV |
| Paired smoke | `run_pi05_paired_replan_smoke.py` | `reproducibility_report.json` |
| Shared-control eval | `run_pi05_paired_replan_node_eval.py` | `experiment_plan.json`, `progress.json`, `paired_summary.json` |
| Optional second round | `make_pi05_round2_no_rescue_nodes.py` | a new node manifest for the no-rescue scenes |
| Outcome plot | `plot_pi05_scene_outcomes.py` | PNG, PDF, and per-scene CSV |
| End-to-end audit | `audit_pi05_replan_pipeline.py` | read-only PASS/FAIL report on stdout |

## 3. Data flow

```text
fixed seed manifest + resolved prompt plaintext
                    |
                    v
     deterministic H50 x six-r traces       600 policy episodes
                    |
                    +--> prompt SHA-256 audit
                    |
                    v
       r25 fail AND another r succeeds
                    |
                    v
   successful-r boundaries -> r25 state map
                    |
                    v
      <=3 window families -> <=10 nodes
                    |
                    v
     3x control + 3x forced paired smoke       6 policy episodes
                    |
                    v
      N shared controls + M forced nodes      N+M policy episodes
                    |
                    v
  paired validity gates + rescue aggregation
                    |
                    v
 r25 baseline + distinct rescued scenes -> scene_outcomes
```

Only a valid pair can contribute a rescue. Repeated successful nodes in one
scene count as one rescued scene in the final scene success rate.

## 4. Standard variables and preflight

Use a new output directory for every task/run. Do not use `--reset` on an
audit artifact.

```bash
cd /home/ubuntu/Workspace/RoboTwin

TASK=<demo_clean_task>
OUT=/home/ubuntu/Workspace/RoboTwin/eval_result/${TASK}/pi05_remote/<fresh_run_name>
CHECKPOINT=/home/ubuntu/Model/pi0.5_robotwin2
SERVER_CONFIG=pi05_robotwin2_multitask_pytorch
SERVER_GPU=<free_server_gpu>
CLIENT_GPU=<free_client_gpu>
PORT=<free_port>
R0=25

nvidia-smi
ss -ltnp | grep ":${PORT} " || true
test ! -e "$OUT"
```

The final `test` must succeed before a new discovery run. If resuming the exact
same run, inspect `grid_results.json`, logs, process list, port, and GPUs before
relaunching.

## 5. Stage A: deterministic 600-episode discovery grid

```bash
.venv/bin/python script/grid_search_pi05_hr.py \
  --h-values 50 \
  --r-values 10 15 25 30 35 40 \
  --eval-num 100 \
  --checkpoint-step 15000 \
  --checkpoint-dir "$CHECKPOINT" \
  --server-config "$SERVER_CONFIG" \
  --model-name pi0.5_robotwin2 \
  --task-name "$TASK" \
  --task-config demo_clean \
  --seed 0 \
  --server-gpu "$SERVER_GPU" \
  --client-gpu "$CLIENT_GPU" \
  --port "$PORT" \
  --deterministic-torch \
  --wandb-mode disabled \
  --trace-root "$OUT/traces" \
  --resolved-manifest "$OUT/resolved_episode_manifest.json" \
  --output-dir "$OUT"
```

Acceptance contract:

- 6 configurations and exactly 100 traces in each;
- the ordered 100 scene seeds are identical for every r;
- trace instruction equals resolved plaintext;
- every chunk fingerprint equals SHA-256 of that plaintext;
- server log confirms deterministic Torch.

Recover and independently audit the prompt plaintext:

```bash
.venv/bin/python script/recover_pi05_prompt_manifest.py \
  --seed-manifest "$OUT/seed_manifest_seed0_eval100.json" \
  --trace-root "$OUT/traces" \
  --output "$OUT/recovered_resolved_episode_manifest.json" \
  --report "$OUT/prompt_recovery_report.json" \
  --instruction-type unseen \
  --max-attempts 3
```

Require `all_exactly_recovered=true`.

## 6. Stage B: contrast scenes, mapped windows, and nodes

```bash
SUBSET="$OUT/seed_manifest_r25_fail_other_success.json"
SUBSET_PROMPTS="$OUT/resolved_episode_manifest_r25_fail_other_success.json"
CONTROLS="$OUT/replan_window_controls_r25_fail_other_success.json"
NODES="$OUT/replan_nodes_r25_fail_other_success.json"

.venv/bin/python script/make_pi05_outcome_subset_manifest.py \
  --source-manifest "$OUT/seed_manifest_seed0_eval100.json" \
  --trace-root "$OUT/traces" \
  --required-failure-r "$R0" \
  --output "$SUBSET" \
  --source-resolved-manifest "$OUT/resolved_episode_manifest.json" \
  --output-resolved-manifest "$SUBSET_PROMPTS"

.venv/bin/python script/build_pi05_replan_window_controls.py \
  --r0 "$R0" \
  --scene-success-map "$SUBSET" \
  --trace-root "$OUT/traces" \
  --horizon 50 \
  --delta 2 \
  --window-radius 2 \
  --state-lookback 5 \
  --max-state-distance 0.35 \
  --max-windows-per-scene 3 \
  --output "$CONTROLS"

.venv/bin/python script/extract_pi05_replan_nodes.py \
  --controls "$CONTROLS" \
  --max-nodes-per-scene 10 \
  --output "$NODES" \
  --output-csv "$OUT/replan_nodes_r25_fail_other_success.csv"
```

These three scripts consume existing traces and run zero policy episodes.
Review scenes with zero candidate windows explicitly; do not silently remove
them from the reporting population.

## 7. Stage C: paired reproducibility smoke

Choose a scene/node from `NODES` that the r25 failure reaches.

```bash
SMOKE_SEED=<seed>
SMOKE_NODE=<node>
SMOKE_OUT="$OUT/paired_smoke_r25_scene${SMOKE_SEED}_after${SMOKE_NODE}"

.venv/bin/python script/run_pi05_paired_replan_smoke.py \
  --source-seed-manifest "$SUBSET" \
  --source-resolved-manifest "$SUBSET_PROMPTS" \
  --output-dir "$SMOKE_OUT" \
  --checkpoint-dir "$CHECKPOINT" \
  --server-config "$SERVER_CONFIG" \
  --model-name pi0.5_robotwin2 \
  --task-name "$TASK" \
  --task-config demo_clean \
  --scene-seed "$SMOKE_SEED" \
  --node "$SMOKE_NODE" \
  --repeats 3 \
  --horizon 50 \
  --r0 "$R0" \
  --server-gpu "$SERVER_GPU" \
  --client-gpu "$CLIENT_GPU" \
  --port "$PORT" \
  --wandb-mode disabled
```

Do not continue unless `reproducibility_report.json` has `passed=true`.
This stage costs exactly 6 policy episodes.

## 8. Stage D: shared-control paired evaluation

First materialize and inspect the immutable experiment plan:

```bash
PAIRED_OUT="$OUT/paired_replan_node_eval_r25_deterministic"

.venv/bin/python script/run_pi05_paired_replan_node_eval.py \
  --nodes "$NODES" \
  --seed-manifest "$SUBSET" \
  --resolved-manifest "$SUBSET_PROMPTS" \
  --checkpoint-dir "$CHECKPOINT" \
  --server-config "$SERVER_CONFIG" \
  --model-name pi0.5_robotwin2 \
  --task-name "$TASK" \
  --task-config demo_clean \
  --horizon 50 \
  --r0 "$R0" \
  --server-gpu "$SERVER_GPU" \
  --client-gpu "$CLIENT_GPU" \
  --port "$PORT" \
  --wandb-mode disabled \
  --output-dir "$PAIRED_OUT" \
  --prepare-only
```

Let `N` be the number of selected scenes and `M` the sum of their node counts.
The plan must contain exactly `N` controls and `M` forced cases, for `N+M`
policy episodes. Remove only `--prepare-only` to execute the same plan. The
launcher resumes cases only when their `metrics.json` is absent.

Validity requires:

- same resolved instruction;
- control and forced both reach node `t`;
- bitwise-identical recorded node observation;
- identical attempted targets for actions `1..t`;
- exactly one forced marker at `t+1`;
- replacement inference fingerprint equals the recorded node observation.

## 9. Optional second node round

This is outcome-adaptive follow-up analysis, not part of the frozen primary
candidate test. Store it in a new node manifest and a new paired output
directory. Never append cases to the primary immutable plan.

```bash
.venv/bin/python script/make_pi05_round2_no_rescue_nodes.py --help
```

Run the generated node manifest through the same `--prepare-only` and
shared-control evaluator sequence. Report first-round and second-round results
separately as well as the union of distinct rescued scenes.

## 10. Stage E: scene outcomes

The one-replan row starts from the complete r25 grid row. A baseline r25
failure turns green only if at least one valid pair for that seed is a rescue.
An r25 failure that was not paired, or was paired but not rescued, remains red.

```bash
PLOT_OUT="$OUT/scene_outcome_plot"

.venv/bin/python script/plot_pi05_scene_outcomes.py \
  --trace-root "$OUT/traces" \
  --output-dir "$PLOT_OUT" \
  --paired-summary "$PAIRED_OUT/paired_summary.json" \
  --replan-r 25 \
  --title "${TASK}: outcomes by scene, execution length, and one-replan rescue"
```

For an explicitly reported second round, add another repeatable
`--paired-summary <round2>/paired_summary.json`. Rescue nodes are merged by
seed and printed below the matrix. The scene success rate counts each rescued
seed once, regardless of how many nodes rescue it.

## 11. Read-only end-to-end acceptance

```bash
python3 script/audit_pi05_replan_pipeline.py \
  --root "$OUT" \
  --paired-summary "$PAIRED_OUT/paired_summary.json"
```

Add every reported paired round with another `--paired-summary`. This audit
does not launch GPUs or write into the result directory. Any `FAIL` makes the
process exit non-zero.

## 12. Episode accounting

| Stage | Formula |
|---|---:|
| Discovery | `6 * 100 = 600` |
| Prompt recovery | 0 pi0.5 policy episodes |
| Candidate and node generation | 0 |
| Paired smoke | `3 + 3 = 6` |
| Primary paired eval | `N + M` |
| Optional round k | `N_k + M_k` |
| Plot and audit | 0 |

For the validated `move_playingcard_away` run:

- discovery: 600;
- contrast population: 22 scenes;
- selected window families: 65;
- primary nodes: 129;
- reproducibility smoke: 6;
- primary paired eval: `22 + 129 = 151`;
- optional second round: `11 + 55 = 66`;
- canonical total through primary paired eval: `600 + 6 + 151 = 757`;
- including the optional second round: `823`.

The final union is 73 r25 grid successes plus 14 distinct rescued scenes:
`87/100`. The 39 and 5 rescued **pairs** in the two rounds are node-level
counts and must not be added to the scene-level success rate.
