# pi0.5 RoboTwin replan-node causal evaluation handoff

## Objective and scope

This pipeline asks an exploratory causal question:

> For a scene that fails with a fixed execution length `r0`, can one extra
> replan at a node suggested by successful trajectories with other `r` values
> rescue the episode?

It is not yet a label-free online selector. Success/failure labels from the
six-`r` discovery grid are used to select scenes and candidate nodes. Final
success is validation-only for each paired intervention.

All experiments use:

- RoboTwin: `/home/ubuntu/Workspace/RoboTwin`
- OpenPI: `/home/ubuntu/Workspace/openpi`
- mixed checkpoint:
  [`motus-robotics/pi0.5_robotwin2`](https://huggingface.co/motus-robotics/pi0.5_robotwin2),
  installed at `/home/ubuntu/Model/pi0.5_robotwin2`
- OpenPI config: `pi05_robotwin2_multitask_pytorch`
- task config: `demo_clean`
- fixed predicted horizon: `H=50`
- candidate execution lengths: `r={10,15,25,30,35,40}`
- checkpoint metadata step: `15000`

## 0. Select and freeze a new task

Choose one previously unused `demo_clean` task before examining its policy
outcomes. For a real cross-task replication, do not choose the task because a
tail metric or a particular `r` already looks favorable.

Before evaluation:

1. Inspect `envs/<TASK>.py`, especially `load_actors`, `play_once`, and
   `check_success`.
2. Confirm the mixed checkpoint supports the embodiment and prompt format.
3. Record the task's manipulated object, reference object, required arm,
   release/open-gripper condition, and maximum action count.
4. Use a fresh task-specific output directory. Never copy an output path whose
   `eval_result/<task>` component names another task.
5. Freeze the following before seeing the new outcomes:
   `H=50`, the six `r` values, `r0=25`, `delta=2`, window radius `2`,
   state lookback `5`, maximum state distance `0.35`, at most three window
   families, and at most ten concrete nodes per scene.

Keeping `r0=25` fixed across tasks is cleaner than selecting a task-specific
`r0` after seeing all success labels. If another rule is used, preregister it
and call the result exploratory.

Suggested shell variables:

```bash
cd /home/ubuntu/Workspace/RoboTwin

TASK=<new_task_name>
TASK_CONFIG=demo_clean
OUT=/home/ubuntu/Workspace/RoboTwin/eval_result/${TASK}/pi05_remote/hr_eval100_pi05_robotwin2_deterministic_seed0
CHECKPOINT=/home/ubuntu/Model/pi0.5_robotwin2
SERVER_CONFIG=pi05_robotwin2_multitask_pytorch
SERVER_GPU=2
CLIENT_GPU=3
PORT=8200
```

Use a port/GPU pair not occupied by another evaluation.

## 1. Run the deterministic six-r discovery grid

Deterministic inference must be enabled from the beginning. Do not generate
candidate traces in ordinary CUDA mode and switch to deterministic mode only
for causal validation: deterministic kernels can change the action trajectory
even with the same observation and RNG seed.

```bash
/home/ubuntu/Workspace/RoboTwin/.venv/bin/python \
  script/grid_search_pi05_hr.py \
  --h-values 50 \
  --r-values 10 15 25 30 35 40 \
  --eval-num 100 \
  --checkpoint-step 15000 \
  --checkpoint-dir "$CHECKPOINT" \
  --server-config "$SERVER_CONFIG" \
  --model-name pi0.5_robotwin2 \
  --task-name "$TASK" \
  --task-config "$TASK_CONFIG" \
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

Expected output:

- `$OUT/seed_manifest_seed0_eval100.json`
- `$OUT/resolved_episode_manifest.json`
- `$OUT/grid_results.json`
- `$OUT/trial_000_H50_r10` through `trial_005_H50_r40`
- `$OUT/traces/H50_r*/`

Acceptance checks:

- exactly six complete configurations;
- exactly 100 episodes per configuration;
- identical ordered scene-seed list for every `r`;
- every resolved prompt is non-null;
- every trace has one prompt fingerprint and one episode outcome;
- no task/output-directory mismatch;
- OpenPI logs state `Enabled deterministic PyTorch inference kernels`.

The six-`r` grid is 600 episodes. `r>=10` preserves the serving-budget floor of
at least ten actions per model call.

## 2. Audit prompt plaintext and fingerprints

The resolved prompt manifest is the experimental input. Reconstructing the
same scene is not enough: the prompt text must also be identical.

```bash
/home/ubuntu/Workspace/RoboTwin/.venv/bin/python \
  script/recover_pi05_prompt_manifest.py \
  --seed-manifest "$OUT/seed_manifest_seed0_eval100.json" \
  --trace-root "$OUT/traces" \
  --output "$OUT/recovered_resolved_episode_manifest.json" \
  --report "$OUT/prompt_recovery_report.json" \
  --instruction-type unseen \
  --max-attempts 3
```

Require:

- one SHA-256 prompt fingerprint per scene across all six `r`;
- `all_exactly_recovered=true`;
- recovered plaintext hash equals the trace fingerprint;
- ordered seeds and prompts align with the seed manifest.

If expert replay is unstable, resume the recovery audit; do not replace the
scene or silently accept a different prompt.

## 3. Freeze `r0` and select contrast scenes

Recommended fixed reference: `r0=25`.

Select a scene only when:

1. deterministic `r0=25` failed; and
2. at least one of the other five deterministic `r` trajectories succeeded.

```bash
R0=25
SUBSET="$OUT/seed_manifest_r25_fail_other_success.json"
SUBSET_PROMPTS="$OUT/resolved_episode_manifest_r25_fail_other_success.json"

/home/ubuntu/Workspace/RoboTwin/.venv/bin/python \
  script/make_pi05_outcome_subset_manifest.py \
  --source-manifest "$OUT/seed_manifest_seed0_eval100.json" \
  --trace-root "$OUT/traces" \
  --required-failure-r "$R0" \
  --output "$SUBSET" \
  --source-resolved-manifest "$OUT/resolved_episode_manifest.json" \
  --output-resolved-manifest "$SUBSET_PROMPTS"
```

The subset records, per scene:

- `successful_rs`;
- `failed_rs`;
- the fixed `r0`;
- the original scene index;
- the aligned resolved prompt.

Do not silently drop scenes based on later paired-control outcomes. Report
them as control-success, control-failure, or node-not-reached strata.

## 4. Build candidate replan-window controls

For the failed `r0` trajectory, let its natural boundaries be `R0`. For every
successful `r`, let its natural boundaries be `R+`.

Raw candidate rule:

```text
t in R+ is new only if min(|t-u| for u in R0) > 2 actions
raw window = [t-2, t+2]
```

Each successful boundary is then mapped into the failed trajectory:

- reconstruct the preceding five executed 12-D arm-joint targets;
- omit both grippers;
- find the most similar five-action segment in the failed `r0` trajectory;
- distance is the mean of left/right six-joint L2;
- reject distance above `0.35`;
- reject mappings within two actions of an existing `r0` boundary;
- reject mappings at the failure endpoint;
- merge overlapping or adjacent mapped windows;
- give at most one support vote per distinct successful `r`;
- rank by distinct-`r` support, lower state distance, then earlier time;
- retain at most three window families per scene.

```bash
CONTROLS="$OUT/replan_window_controls_r25_fail_other_success.json"

/home/ubuntu/Workspace/RoboTwin/.venv/bin/python \
  script/build_pi05_replan_window_controls.py \
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
```

Important: this is stage matching by policy joint targets, not measured object
state. A low distance does not prove semantic task-stage equivalence.

## 5. Extract concrete intervention nodes

For each selected window family:

- preserve one representative mapped time;
- add other exact mapped times by distinct successful-`r` support;
- break ties by lower state distance, window rank, and earlier time;
- deduplicate;
- retain at most ten nodes per scene.

```bash
NODES="$OUT/replan_nodes_r25_fail_other_success.json"
NODES_CSV="$OUT/replan_nodes_r25_fail_other_success.csv"

/home/ubuntu/Workspace/RoboTwin/.venv/bin/python \
  script/extract_pi05_replan_nodes.py \
  --controls "$CONTROLS" \
  --max-nodes-per-scene 10 \
  --output "$NODES" \
  --output-csv "$NODES_CSV"
```

Node semantics are exact:

```text
node t = execute actions 1..t, then replan immediately before action t+1
```

The policy API therefore receives `t+1` in
`pi05_force_replan_before_actions`.

## 6. Run a deterministic paired smoke test

Pick one selected scene/node that the failed `r0` trajectory reaches. Run:

- control: no extra replan, three exact repeats;
- forced: one extra replan at the node, three exact repeats.

```bash
SMOKE_SEED=<selected_scene_seed>
SMOKE_NODE=<selected_node>
SMOKE_OUT="$OUT/paired_smoke_r25_scene${SMOKE_SEED}_after${SMOKE_NODE}"

/home/ubuntu/Workspace/RoboTwin/.venv/bin/python \
  script/run_pi05_paired_replan_smoke.py \
  --source-seed-manifest "$SUBSET" \
  --source-resolved-manifest "$SUBSET_PROMPTS" \
  --output-dir "$SMOKE_OUT" \
  --checkpoint-dir "$CHECKPOINT" \
  --server-config "$SERVER_CONFIG" \
  --model-name pi0.5_robotwin2 \
  --task-name "$TASK" \
  --task-config "$TASK_CONFIG" \
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

Require `reproducibility_report.json: passed=true`, meaning:

- all six runs have the same bitwise node observation
  (state plus three RGB inputs);
- all six runs have identical attempted action targets through node `t`;
- each arm is internally identical in chunks, targets, outcome, and execution
  counters;
- forced marker is exactly `t+1`;
- forced replacement chunk uses the recorded node observation.

Smoke success means the causal comparison is reproducible. It does not mean
the intervention improves task success. `control=0/3, forced=0/3` can still be
a valid reproducibility smoke.

## 7. Run the full shared-control paired evaluation

For `N` selected scenes and `M` total nodes, run:

```text
N shared deterministic controls + M forced-node treatments
```

One control can be reused for all nodes of its scene only because determinism
and the prefix/node-observation gate were validated. Total trials are not
necessarily 156 on a new task.

Prepare and inspect the plan without GPUs:

```bash
PAIRED_OUT="$OUT/paired_replan_node_eval_r25_deterministic"

/home/ubuntu/Workspace/RoboTwin/.venv/bin/python \
  script/run_pi05_paired_replan_node_eval.py \
  --nodes "$NODES" \
  --seed-manifest "$SUBSET" \
  --resolved-manifest "$SUBSET_PROMPTS" \
  --checkpoint-dir "$CHECKPOINT" \
  --server-config "$SERVER_CONFIG" \
  --model-name pi0.5_robotwin2 \
  --task-name "$TASK" \
  --task-config "$TASK_CONFIG" \
  --horizon 50 \
  --r0 "$R0" \
  --server-gpu "$SERVER_GPU" \
  --client-gpu "$CLIENT_GPU" \
  --port "$PORT" \
  --wandb-mode disabled \
  --output-dir "$PAIRED_OUT" \
  --prepare-only
```

Then remove only `--prepare-only` and launch. The launcher is resumable and
skips cases that already contain `metrics.json`.

Outputs:

- `experiment_plan.json`
- `progress.json`
- one control directory per scene;
- one forced directory per scene/node;
- `paired_summary.json`

Each one-episode process receives a unique occurrence token. This forces the
server to reset its primary inference index without adding new RNG entropy.

## 8. Final statistics

A forced/control pair is valid only when all gates pass:

- same resolved instruction;
- both trajectories reach node `t`;
- bitwise-identical node observation;
- identical attempted policy targets for actions `1..t`;
- exact forced marker before action `t+1`;
- replacement inference uses the node observation.

Primary outcomes:

- `rescue`: control failure, forced success;
- `harm`: control success, forced failure;
- `both_success`;
- `both_failure`;
- invalid or node-not-reached pair.

Report rescue rate only among valid control-failure pairs. Multiple nodes from
one scene are dependent; aggregate by scene or use scene-clustered inference.
Do not treat all nodes as independent episodes.

## Known pitfalls

1. **Per-episode RNG seeding alone is insufficient.** PyTorch CUDA inference
   produced different chunks for identical observations and seeds until
   deterministic algorithms, deterministic cuDNN, disabled TF32, and
   `CUBLAS_WORKSPACE_CONFIG=:4096:8` were enabled.
2. **Do not switch inference modes between discovery and validation.**
   Deterministic mode changed the policy trajectory from the very first chunk
   in the `place_a2b_left` smoke, invalidating a candidate node derived from
   ordinary-CUDA traces.
3. **Forced-only evaluation is not causal.** Every treatment needs a normal
   control with an identical prefix and node observation.
4. **Off-by-one node semantics matter.** Candidate `t` is after action `t`;
   intervention is before one-based action `t+1`.
5. **Repeated scene seeds need an occurrence reset token.** Otherwise separate
   one-episode client processes can inherit the server's previous inference
   index even though the scene seed is unchanged.
6. **Prompt text is part of the scene.** Lock plaintext in a resolved manifest
   and audit its SHA-256 against every trace.
7. **Never silently substitute unstable manifest scenes.** Expert replay drift
   is diagnostic; selective replacement changes the evaluation population.
8. **Candidate generation is label-assisted and exploratory.** A new task is a
   replication of the mechanism, not proof of a deployable label-free online
   selector.
9. **The stage mapper uses action targets, not actual object state.** Similar
   arm targets can correspond to different objects or task phases.
10. **Vote by distinct successful `r`, not rollout count.** One `r` gets at
    most one support vote per window.
11. **SAPIEN contact enumeration order is not stable.** Sort contact pairs by
    content before exact trace hashing; do not confuse list order with physics
    nondeterminism.
12. **TOPP success does not imply task success.** Inspect object identity,
    contact timing, release, final gripper state, and the task's explicit
    `check_success`.
13. **A selected node may not be reached.** Early episode success/termination
    makes the pair invalid rather than a failed intervention.
14. **Do not reuse the old 63 forced-only trials.** They were generated without
    deterministic Torch and without paired controls.
15. **Avoid duplicate launchers.** Before launching, inspect the port, process
    list, GPUs, and output directory. Two launchers writing one directory can
    corrupt results.
16. **Use a fresh output directory and avoid `--reset`.** Original evaluation
    inputs/results are immutable audit artifacts.
17. **Current trace filenames include occurrence suffixes.** Candidate scripts
    now accept both old `scene_<seed>.json` and new
    `scene_<seed>_episode_<id>.json`.

## Reference: current `place_a2b_left` run

Discovery grid, ordinary CUDA (therefore not reusable for the corrected
cross-task procedure):

```text
r10: 59/100
r15: 76/100
r25: 73/100
r30: 63/100
r35: 55/100
r40: 56/100
```

With `r0=25`, 22 scenes failed at `r0` and succeeded for at least one other
`r`; three window families and at most ten nodes per scene produced 134 nodes.
The corrected deterministic paired plan is 22 controls + 134 forced = 156.

The smoke at `scene_seed=100000, node=30` passed exact reproducibility but
returned `control=0/3, forced=0/3`. The deterministic forced trajectory moved
toward the wrong object and never reopened the left gripper. This exposed the
discovery/validation inference-mode mismatch above.
