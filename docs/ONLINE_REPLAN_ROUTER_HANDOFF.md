# Online replan-router handoff

Last updated: 2026-07-27 UTC

This document hands off the completed `Move Playingcard Away` offline-router
and online-control experiments. It supplements, rather than replaces, the
forced-node causal pipeline handoff:

- [`PI05_REPLAN_CAUSAL_PIPELINE_HANDOFF.md`](PI05_REPLAN_CAUSAL_PIPELINE_HANDOFF.md)
- [`PI05_REPLAN_PIPELINE_RUNBOOK.md`](PI05_REPLAN_PIPELINE_RUNBOOK.md)
- [`V0_THRESHOLD_SWEEP_HANDOFF.md`](V0_THRESHOLD_SWEEP_HANDOFF.md)

## 1. Current state

The minimum router is implemented, trained, and connected to online RoboTwin
control without modifying pi0.5 weights. Three online evaluations are
complete:

| Experiment | Episodes | Valid pairs | Control | Router | Rescue | Harm |
|---|---:|---:|---:|---:|---:|---:|
| seed100006 smoke, three repeats | 6 | 3/3 | 0/3 | 3/3 | 3 | 0 |
| 14 held-out dataset scenes | 28 | 14/14 | 11/14 | 12/14 | 1 | 0 |
| 100 new expert-valid seeds | 200 | 100/100 | 81/100 | 81/100 | 7 | 7 |

The 100-scene result is the most relevant generalization result. Querying
every two actions with up to three replans produced no net success gain:

```text
control success = 81%
router success  = 81%
rescues         = 7
harms           = 7
```

A reset-safe V0 threshold correction is currently running detached as PID
`4064479`. Its live state, failure diagnosis, output paths, and completion
procedure are in [`V0_THRESHOLD_SWEEP_HANDOFF.md`](V0_THRESHOLD_SWEEP_HANDOFF.md).

## 2. Non-negotiable workspace rules

All continuing code changes belong in:

```text
/home/ubuntu/Workspace/Event-triggered-replanning-for-VLA-control
```

Do not edit the upstream checkout:

```text
/home/ubuntu/Workspace/RoboTwin
```

The upstream RoboTwin and OpenPI trees are runtime dependencies only. The
online launcher prepends the repository's policy overlay to `PYTHONPATH`, so
RoboTwin imports this repository's `pi05_remote.py` without copying it into
RoboTwin.

Do not overwrite an existing output directory. The launchers write immutable
plans/manifests and refuse incompatible reuse. Use a new output root for every
protocol change.

## 3. External dependencies

Validated paths:

```text
RoboTwin:   /home/ubuntu/Workspace/RoboTwin
OpenPI:     /home/ubuntu/Workspace/openpi
checkpoint: /home/ubuntu/Model/pi0.5_robotwin2
```

Checkpoint source:

<https://huggingface.co/motus-robotics/pi0.5_robotwin2>

Python environments:

```text
/home/ubuntu/Workspace/RoboTwin/.venv/bin/python
/home/ubuntu/Workspace/openpi/.venv/bin/python
```

The server needs the OpenPI environment. RoboTwin simulation and launchers use
the RoboTwin environment.

## 4. Core implementation files

| File | Role |
|---|---|
| `robotwin/script/train_replan_router.py` | Build the paired dataset, extract frozen pi0.5 features, train the two-head MLP, and evaluate scene-grouped splits |
| `openpi/scripts/serve_robotwin_router_policy.py` | Serve ordinary pi0.5 action chunks and side-effect-free router queries from one frozen model process |
| `robotwin/policy/pi05_remote.py` | Query the router inside RoboTwin, cut stale chunk tails, enforce absolute cadence, max-replan budget, and cooldown |
| `robotwin/script/run_pi05_online_router_paired_eval.py` | Build immutable paired plans, run control/router episodes, and audit prefixes, observations, triggers, replacements, cadence, and outcomes |
| `robotwin/script/select_robotwin_unseen_scenes.py` | Select expert-valid scene seeds absent from discovered prior task manifests/features and freeze prompt plaintext |
| `temp/run_move_playingcard_unseen100_online_eval.sh` | Exact completed 100-scene selection/evaluation wrapper |

Generated data and checkpoints are under `temp/`, which is intentionally
gitignored. A clone of the GitHub repository will not contain the experiment
artifacts described below.

## 5. Offline dataset and router

### 5.1 Dataset

The data comes from the existing absolute-cadence, shared-prefix
counterfactual evaluations. Each decision sample contains:

```text
scene seed
post-action decision clock t
observation at t
old [50, 14] action chunk
old chunk cursor
keep_success
replan_success
```

Both-failure samples are excluded from the first version. The final dataset
contains:

```text
348 samples
89 scenes
56 rescue samples
276 both-success samples
16 harm samples
128 excluded both-failure samples
```

The split is deterministic and grouped by scene:

```text
train: 242 samples / 62 scenes
val:    54 samples / 13 scenes
test:   52 samples / 14 scenes
```

No candidate nodes from one scene cross splits.

Each sample is recovered from a validated forced-replan trace, not inferred
from filenames alone:

1. find the unique marker that cuts the old chunk before action `t+1`;
2. read the full old `[50,14]` chunk from the marker's chunk record;
3. read `previous_chunk_cursor` from the following replacement chunk;
4. load the archived post-action observation at `t`;
5. require its fingerprint to equal both the paired-summary node fingerprint
   and the replacement chunk's inference fingerprint;
6. require the prompt plaintext hash to equal the chunk trace hash;
7. reject duplicate `(task, scene_seed, t)` samples across roots.

The old cursor is archived for integrity and analysis, but the current router
does **not** encode it. Likewise, the selected `vision_encoder + vision`
configuration has no direct timestep, proprioceptive state, prompt, or action
input.

### 5.2 Features

Two frozen visual representations are implemented:

1. `vision_encoder`
   - apply the normal pi0.5 input transform and observation preprocessing;
   - call `paligemma_with_expert.embed_image` independently for each camera;
   - mean-pool the projected SigLIP tokens over the token dimension;
   - multiply by the camera-presence mask;
   - concatenate high, left-wrist, and right-wrist vectors;
   - tensor flow: three `[B,2048]` vectors to `z_v:[B,6144]`.
2. `vlm_hidden`
   - build the full image/language prefix with `embed_prefix`;
   - run the frozen PaliGemma backbone with the ordinary 2-D/4-D attention
     masks and position IDs;
   - select the final hidden states at language-token positions;
   - masked-mean over valid language tokens;
   - tensor flow: language hidden `[B,L,2048]` to `z_v:[B,2048]`.

Both extractors run under `torch.inference_mode()`. pi0.5 is set to `eval`,
every parameter has `requires_grad=False`, and no action diffusion sampling is
performed during offline extraction.

The image tensor is not fed to an independently trained CNN. These features
come from the published pi0.5 visual/VLM modules after the same policy
transforms used by online inference.

The online server repeats the transformed sample to extraction batch size
four. This matches the CUDA kernel shape used by offline feature extraction.
On archived seed100006 at `t=37`, online/offline `p_keep` differed by only
`2.87e-5`, with the same trigger decision.

### 5.3 Router

The router predicts two outcome logits:

```text
p_keep   = sigmoid(keep_success_logit)
p_replan = sigmoid(replan_success_logit)
trigger iff p_replan - p_keep > lambda
```

The selected model is:

```text
feature_type: vision_encoder
router_input: vision
hidden_dim: 128
outputs: keep_success, replan_success
parameters: 786,818
lambda: 0.05
best epoch: 2
```

#### 5.3.1 Tensor architecture

The selected visual-only network is:

```text
z_v [B, 6144]
  -> Linear(6144, 128)
  -> GELU
  -> Dropout(p=0.1)
  -> Linear(128, 2)
  -> [keep_logit, replan_logit]
```

Its parameter count is:

```text
6144*128 + 128 + 128*2 + 2 = 786,818
```

The `vision_action` variant adds:

```text
old chunk [B, 50, 14]
  -> flatten [B, 700]
  -> Linear(700, 128)
  -> GELU
  -> Linear(128, 64)
  -> GELU
  -> z_a [B, 64]

concat(z_v, z_a)
  -> Linear(visual_dim + 64, 128)
  -> GELU
  -> Dropout(p=0.1)
  -> Linear(128, 2)
```

The `vision_encoder + vision_action` variant has 892,994 parameters. The
action encoder consumes the complete original chunk, including its already
executed prefix; it does not crop the input at `old_chunk_cursor`.

#### 5.3.2 Supervision

There is no single training label modified by lambda. The target for every
sample is a two-element Bernoulli vector:

| Counterfactual transition | Target `[keep, replan]` |
|---|---|
| rescue | `[0,1]` |
| harm | `[1,0]` |
| both success | `[1,1]` |
| both failure | excluded |

Training minimizes the unweighted mean binary cross-entropy over both logits:

```text
loss = BCEWithLogits([keep_logit, replan_logit],
                     [keep_success, replan_success])
```

There are no class weights, focal loss, balanced sampler, oversampling, or
explicit replan-cost term in the optimizer objective. The strong-replan class
used for ROC/F1 metrics is only:

```text
keep_success == 0 and replan_success == 1
```

`lambda=0.05` is applied after both probabilities are predicted. Changing
lambda does not require retraining and does not alter saved labels.

#### 5.3.3 Normalization

Statistics are computed from training samples only:

```text
visual_mean/std: independently for every z_v dimension
action_mean/std: independently for each of 14 action dimensions after
                 flattening all train scenes, chunks, and horizon positions
```

Statistics accumulate in float64 and are saved as float32. Any standard
deviation below `1e-6` is replaced by one. The same tensors are embedded in
the router checkpoint and applied by the online server.

Although action normalization is computed for every run, it has no effect on
the selected `router_input=vision` forward pass.

#### 5.3.4 Scene split

Scenes are first assigned to exactly one of three strata:

```text
has_rescue
has_harm
both_success_only
```

Within each stratum, scene seeds are deterministically shuffled using
`split_seed + 10007*(stratum_index+1)` and divided approximately 70/15/15.
The split assertions require:

- no scene overlap;
- every scene assigned exactly once;
- non-empty train/validation/test sets;
- both strong-replan and non-strong-replan samples in every split.

This is scene-grouped but not task-instance-grouped beyond the seed: all data
belongs to `move_playingcard_away`.

#### 5.3.5 Optimization and checkpoint selection

The completed runs used:

```text
optimizer: AdamW
learning rate: 1e-3
weight decay: 1e-4
train batch size: 32
maximum epochs: 200
early-stopping patience: 30
train seed: 0
split seed: 0
feature extraction batch size: 4
dropout: 0.1
```

Training samples are reshuffled each epoch with a CPU
`torch.Generator(seed=0)`. There is no learning-rate scheduler and no gradient
clipping.

After every epoch, model selection uses validation ROC-AUC of:

```text
score = p_replan - p_keep
positive = rescue
negative = harm or both success
```

Higher validation ROC-AUC wins. Exact AUC ties are broken by validation
counterfactual utility at `lambda=0.05`. The selected state is cloned to CPU;
the test split is evaluated only after selection. For the primary model,
epoch 2 was best and early stopping ended after 32 epochs.

The reported `decision_temperature=0.10` only produces an auxiliary smooth
decision probability for diagnostics. It is not used by the model, loss,
hard trigger, or checkpoint selection.

#### 5.3.6 Determinism and saved checkpoint

Before extraction/training the script:

```text
sets Python, NumPy, CPU Torch, and CUDA Torch seeds
enables torch deterministic algorithms
sets cuDNN deterministic and disables benchmarking
disables TF32 for CUDA matmul and cuDNN
uses CUBLAS_WORKSPACE_CONFIG=:4096:8
```

The checkpoint stores:

- router state dict;
- complete architecture metadata;
- visual/action normalization tensors;
- feature archive path and SHA-256;
- dataset fingerprint and complete scene split;
- lambda, best epoch, and train seed.

The optional action encoder reduced held-out performance in this dataset.
The selected checkpoint is therefore visual-only.

Test results:

| Features | Input | ROC-AUC | Accuracy | F1 | Strong-replan recall |
|---|---|---:|---:|---:|---:|
| vision encoder | vision | 0.8977 | 0.9231 | 0.7778 | 0.875 |
| vision encoder | vision+action | 0.8750 | 0.7885 | 0.5217 | 0.750 |
| VLM hidden | vision | 0.5540 | 0.6154 | 0.2308 | 0.375 |
| VLM hidden | vision+action | 0.6193 | 0.6923 | 0.2727 | 0.375 |

Primary offline artifacts:

```text
temp/outputs/replan_router_minimal_validation/
├── features_vision_encoder.npz
├── features_vision_encoder_manifest.json
├── router_vision_encoder_vision.pt
└── evaluation_vision_encoder_vision.json
```

Primary artifact hashes:

```text
features NPZ:
4150b1092f36dff40c6c00c15b2d5f154c1caa009e0d2e1a787ce753fbbef071

feature manifest:
2d7bf1488e3a613b7d35d2550a1367aa6c9e44570110ba06724a059e17daa806

router checkpoint:
2aa0fb6070b4509b22a5bbc5dcb5cfffefc4d6a07321742c165301581ffae242
```

The directory
`superseded_vision_encoder_sum_pool/` contains an earlier incompatible
feature experiment. Do not use it.

## 6. Online-control semantics

### 6.1 Decision clock and off-by-one

A router node `t` means:

1. actions `1..t` have completed;
2. read the post-action observation;
3. score the currently executing old chunk;
4. if triggered, discard its unused tail;
5. infer a replacement chunk before one-based action `t+1`.

The router query itself does not sample actions and does not advance the
executed VLA RNG stream.

### 6.2 Absolute r0 cadence

`r0=25` remains anchored to:

```text
25, 50, 75, ...
```

For example, a trigger at `t=16` executes nine replacement actions to the next
boundary at 25. It does not start a new 25-action relative clock.

### 6.3 Completed 100-scene protocol

```text
H: 50
r0: 25
lambda: 0.05
query interval: every 2 completed actions
last query clock: 398
natural r0 boundaries excluded
maximum router replans: 3
minimum distance between router replans: 25 completed actions
deterministic Torch: enabled
```

Eligible clocks are:

```text
2, 4, 6, ..., 398
```

but `50, 100, 150, ...` are excluded because the ordinary controller already
replans at those natural boundaries.

After a trigger at `t_i`, router queries are suppressed until:

```text
t - t_i >= 25
```

Because the interval grid is even, a trigger at `t=10` can next be queried at
`t=36`, not 35.

## 7. Online experiments

### 7.1 Three-repeat smoke

Output:

```text
temp/outputs/move_playingcard_away_online_router_smoke_seed100006_v2/
```

Protocol:

```text
seed: 100006
3 control + 3 router episodes
candidate-gated
at most one replan
```

All three router runs triggered at `t=37`. Every pair passed seed, prompt,
fingerprint, prefix-target, replacement-observation, and absolute-cadence
checks. All three transitions were rescue.

This smoke validates the original one-replan online integration. It does not
validate three-replan reproducibility.

### 7.2 Fourteen held-out scenes

Output:

```text
temp/outputs/move_playingcard_away_online_router_heldout_test_v2/
```

Result:

```text
14/14 valid pairs
control: 11/14 = 78.57%
router:  12/14 = 85.71%
rescues: 1
harms:   0
```

Five scenes triggered. Seed100006 triggered at `t=37` and was rescued.
Seeds100022 and 100032 triggered at `t=15` and remained successful.
Seeds100044 and 100049 triggered too early and remained failures.

This was candidate-gated. Three failure scenes used their full prior
counterfactual candidate lists; the other eleven used `[10, 15, 30, 40]`.
It is not an unseen-scene estimate.

### 7.3 One hundred new scenes

Selection output:

```text
temp/outputs/move_playingcard_away_online_router_unseen100/inputs/
├── unseen_seed_manifest.json
├── unseen_resolved_episode_manifest.json
└── unseen_selection_audit.json
```

The definition of unseen is seed-level:

- absent from all discovered `move_playingcard_away` seed/resolved manifests
  under RoboTwin `eval_result`;
- absent from project `temp` task manifests;
- absent from the router feature archive at selection start.

One hundred expert-valid seeds were selected from the sequence beginning at
200000. Seeds200080 and 200081 were rejected and replaced, leaving 100 unique
selected seeds through 200101.

Selection checks:

```text
selected_count: 100
selected_unique: true
selected_disjoint_from_seen: true
previously seen seed count: 100
```

Actual evaluation output:

```text
temp/outputs/move_playingcard_away_online_router_unseen100/
└── eval_max3_interval25/
    ├── experiment_plan.json
    ├── progress.json
    ├── online_router_summary.json
    ├── control/
    └── router/
```

Do not use the sibling `eval/` directory. It is a preserved, superseded
max-one-replan plan stopped at `0/200` before any episode ran.

Final result:

```text
episodes: 200/200
valid pairs: 100/100
invalid pairs: 0

control successes: 81
router successes: 81
rescues: 7
harms: 7
both success: 74
both failure: 12
```

Trigger behavior:

```text
scenes with any trigger: 75
total router replans: 204
mean replans per scene: 2.04

0 replans: 25 scenes
1 replan:   7 scenes
2 replans:  7 scenes
3 replans: 61 scenes

first-trigger minimum: 2
first-trigger median: 16
first-trigger mean: 20.83
first-trigger maximum: 116
```

All 19 control failures used all three router replans:

```text
7 were rescued
12 remained failures
```

Seven initially successful scenes were harmed. All zero-, one-, and
two-trigger scenes remained successful; every rescue, harm, and both-failure
case occurred among scenes that exhausted the three-trigger budget.

If the cost is charged per actual replan:

```text
mean utility = 0.81 - 2.04 * 0.05 = 0.708
always-keep utility = 0.81
```

## 8. What the audits establish

For every completed pair, the summary checks:

- identical scene seed;
- identical prompt plaintext and SHA-256;
- router queries only at preregistered clocks;
- recorded query observation equals the observation scored by the server;
- control/router observation and action-target prefix match through the first
  trigger;
- trigger count does not exceed the budget;
- no query occurs inside the 25-action cooldown;
- every trigger has exactly one trace marker;
- every replacement inference uses the trigger observation;
- every replacement ends at the next absolute-r0 boundary;
- no-trigger router trajectories exactly equal their controls.

After the first router trigger, the router trajectory is expected to diverge
from control. Therefore later triggers are not independent shared-prefix
counterfactual trials. Their observations, replacements, cadence, and cooldown
are audited, but the final rescue/harm label belongs to the complete
multi-replan policy versus control, not to an individual later trigger.

## 9. Exact reproduction commands

### 9.1 Train into a fresh directory

Use the OpenPI interpreter:

```bash
cd /home/ubuntu/Workspace/Event-triggered-replanning-for-VLA-control

/home/ubuntu/Workspace/openpi/.venv/bin/python \
  robotwin/script/train_replan_router.py \
  --task move_playingcard_away \
  --feature-type vision_encoder \
  --router-input vision \
  --checkpoint-dir /home/ubuntu/Model/pi0.5_robotwin2 \
  --output-dir /absolute/new/output/directory
```

The script refuses to overwrite an existing router run.

### 9.2 Select another unseen scene set

Run from the RoboTwin checkout:

```bash
cd /home/ubuntu/Workspace/RoboTwin

export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED=0
export PYTHONPATH="/home/ubuntu/Workspace/RoboTwin/envs_invent/curobo/src${PYTHONPATH:+:$PYTHONPATH}"

.venv/bin/python \
  /home/ubuntu/Workspace/Event-triggered-replanning-for-VLA-control/robotwin/script/select_robotwin_unseen_scenes.py \
  --output-dir /absolute/new/input/directory \
  --task-name move_playingcard_away \
  --task-config demo_clean \
  --count 100 \
  --candidate-start 300000 \
  --policy-seed 0 \
  --instruction-type unseen
```

Use a new candidate range and output directory. The unseen audit is evaluated
against manifests visible at selection time.

### 9.3 Run the completed every-two-actions protocol

```bash
cd /home/ubuntu/Workspace/Event-triggered-replanning-for-VLA-control

/home/ubuntu/Workspace/RoboTwin/.venv/bin/python \
  robotwin/script/run_pi05_online_router_paired_eval.py \
  --output-dir /absolute/new/eval/directory \
  --scene-manifest /absolute/new/input/directory/unseen_seed_manifest.json \
  --resolved-scene-manifest /absolute/new/input/directory/unseen_resolved_episode_manifest.json \
  --router-query-interval 2 \
  --router-query-max-action 398 \
  --router-max-replans 3 \
  --router-min-replan-interval 25 \
  --repeats 1 \
  --parallel-groups 4 \
  --server-gpu 0,1,2,3 \
  --client-gpu 0,1,2,3 \
  --port 8400
```

For a long run, start it under tmux. The completed wrapper is:

```text
temp/run_move_playingcard_unseen100_online_eval.sh
```

Change its output root before another experimental run.

## 10. Interpretation and next experiment

The online mechanics are working: all causal/audit checks passed, and the
14-scene candidate-gated evaluation produced a real rescue without harm.
However, the unseen 100-scene result shows that the present router is not a
useful every-two-actions policy:

1. it triggers in 75% of scenes;
2. 61% of scenes exhaust all three replans;
3. many first triggers occur at `t=2..20`, outside the main offline candidate
   distribution;
4. seven rescues are exactly cancelled by seven harms;
5. replan cost makes it substantially worse than always keep.

The next experiment should not simply tune lambda on these 100 test outcomes
and report the same set as an unbiased test. Preserve this set as a completed
test. Recommended next steps:

1. collect new counterfactual labels at interval-grid clocks, especially
   `t=2..24`;
2. add explicit chunk position/cursor and time-since-last-replan features;
3. include a calibrated abstain or minimum-confidence rule;
4. select lambda and cooldown on train/validation scenes only;
5. reserve a fresh seed range, for example starting at 300000, for the next
   final test;
6. repeat a multi-replan smoke three times before another 100-scene run.

Do not interpret the seven rescue nodes as a `+7%` improvement. The seven harms
must be included, and the observed net change is zero.

## 11. Repository and artifact status

At handoff time:

```text
branch: agent/add-checkpoint-source
HEAD: c75f6758536a4d7f78d2b45d1e1e0218aade5071
remote: git@github.com:z-R-rr/Event-triggered-replanning-for-VLA-control.git
```

The online-router work is not committed at that HEAD. Important working-tree
changes include:

```text
M  README.md
M  robotwin/policy/pi05_remote.py
?? openpi/scripts/serve_robotwin_router_policy.py
?? robotwin/script/run_pi05_online_router_paired_eval.py
?? robotwin/script/select_robotwin_unseen_scenes.py
?? robotwin/script/train_replan_router.py
```

There are also pre-existing/user-owned changes and untracked files, including
`robotwin/script/run_pi05_paired_replan_node_eval.py`,
`robotwin/script/plot_eval100_scene_outcomes.py`, `.gitignore`, and `temp/`.
Inspect `git diff` and stage intentionally. Do not revert or overwrite them.

The primary offline artifacts occupy about 22 MiB. The unseen100 online output
occupies about 3.5 GiB. Because `temp/` is ignored, pushing source changes will
not publish those results.
