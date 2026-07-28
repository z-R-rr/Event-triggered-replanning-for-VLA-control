# Event-triggered replanning for VLA control

This repository is a compact, code-only overlay for the deterministic pi0.5
RoboTwin one-replan causal evaluation pipeline. It contains the runtime hooks,
candidate-node construction, shared-control paired evaluation, integrity audit,
and scene-outcome visualization used in the validated
`move_playingcard_away` experiment.

It intentionally does **not** contain RoboTwin/OpenPI source trees, model
weights, evaluation results, videos, logs, or W&B artifacts.

## What the pipeline measures

For a fixed scene that fails with execution length `r0=25`, the pipeline asks:

> Can exactly one extra inference at a selected action boundary rescue the
> episode, while keeping the scene, prompt, random seed, observation, and
> pre-intervention action prefix identical to a shared control?

The frozen protocol uses:

- checkpoint:
  [`motus-robotics/pi0.5_robotwin2`](https://huggingface.co/motus-robotics/pi0.5_robotwin2),
  installed at `/home/ubuntu/Model/pi0.5_robotwin2`;
- predicted horizon: `H=50`;
- discovery grid: `r={10,15,25,30,35,40}`;
- 100 fixed scenes, hence 600 discovery episodes;
- deterministic Torch from discovery onward;
- fixed `r0=25`;
- at most three candidate-window families and ten nodes per scene;
- a 3-control/3-forced reproducibility smoke before full evaluation;
- one shared deterministic control per scene plus one forced run per node.

The full rationale and exact commands are in:

- [`docs/PI05_REPLAN_CAUSAL_PIPELINE_HANDOFF.md`](docs/PI05_REPLAN_CAUSAL_PIPELINE_HANDOFF.md)
- [`docs/PI05_REPLAN_PIPELINE_RUNBOOK.md`](docs/PI05_REPLAN_PIPELINE_RUNBOOK.md)
- [`docs/ONLINE_REPLAN_ROUTER_HANDOFF.md`](docs/ONLINE_REPLAN_ROUTER_HANDOFF.md)

## Repository layout

```text
.
├── robotwin/                  # overlay onto a RoboTwin checkout
│   ├── envs/
│   ├── policy/
│   └── script/
├── openpi/                    # overlay onto an OpenPI checkout
│   ├── scripts/
│   └── src/openpi/training/
├── docs/
└── LICENSES/
```

### RoboTwin runtime files

| File | Purpose |
|---|---|
| `robotwin/envs/_base_task.py` | Deterministic scene/prompt plumbing and trace-compatible task execution hooks |
| `robotwin/policy/pi05_remote.py` | Remote pi0.5 client, compact action/chunk traces, prompt fingerprints, observation recording, occurrence tokens, and forced replan semantics |
| `robotwin/policy/pi05/deploy_policy.yml` | pi0.5 remote-policy deployment options |
| `robotwin/script/eval_policy_wandb.py` | Fixed seed manifest, resolved prompt plaintext, and evaluation entry point |
| `robotwin/script/grid_search_pi05_hr.py` | Deterministic six-r discovery launcher and resumable grid collection |

### Candidate selection and paired evaluation

| File | Purpose |
|---|---|
| `recover_pi05_prompt_manifest.py` | Independently recover prompt plaintext and verify its SHA-256 fingerprint |
| `make_pi05_outcome_subset_manifest.py` | Select exactly the `r0`-fail/other-r-success contrast scenes |
| `build_pi05_replan_window_controls.py` | Map successful-r replan boundaries into the failed-r0 trajectory using five-action, 12-D arm-joint state proxies |
| `extract_pi05_replan_nodes.py` | Rank, deduplicate, and cap concrete intervention nodes |
| `run_pi05_paired_replan_smoke.py` | Run three exact control and three exact forced repeats and enforce reproducibility gates |
| `run_pi05_paired_replan_node_eval.py` | Run one shared control per scene plus all forced-node treatments |
| `run_pi05_paired_replan_smoke_absolute_cadence.py` | Absolute-cadence 3+3 smoke entry point; enables the extra cadence validity gates |
| `run_pi05_paired_replan_node_eval_absolute_cadence.py` | Absolute-cadence shared-control paired-evaluation entry point |
| `make_pi05_round2_no_rescue_nodes.py` | Optional outcome-adaptive second node round for first-round no-rescue scenes |
| `merge_pi05_replan_node_manifests.py` | Build an immutable, provenance-preserving union of candidate nodes from multiple rounds |
| `train_replan_router.py` | Offline frozen-pi0.5 feature extraction and sub-1M-parameter outcome-router training/evaluation |
| `plot_pi05_scene_outcomes.py` | Render per-scene grid outcomes and the r25-plus-one-replan row |
| `audit_pi05_replan_pipeline.py` | Read-only end-to-end integrity audit |

All paths in the table above are under `robotwin/script/` unless otherwise
shown.

### OpenPI server files

| File | Purpose |
|---|---|
| `openpi/scripts/serve_robotwin_policy.py` | Episode-seeded inference, occurrence-token reset, isolated shadow RNG, and deterministic Torch kernel mode |
| `openpi/src/openpi/training/config.py` | `pi05_robotwin2_multitask_pytorch` model/data transform configuration |

These server-side files are required. Installing only the RoboTwin overlay is
not sufficient for causal prefix reproducibility.

## Installation

### 1. Prepare the upstream checkouts

The overlay was validated against:

- RoboTwin commit `c3ddfa8`;
- OpenPI commit `729ac3ecb66f4685f62ab72d77d54d136eaef6fb`.

Install RoboTwin, OpenPI, CUDA, SAPIEN/CuRobo, and their Python environments
according to their upstream documentation. A typical workspace is:

```text
/home/ubuntu/Workspace/RoboTwin
/home/ubuntu/Workspace/openpi
/home/ubuntu/Model/pi0.5_robotwin2
```

The mixed checkpoint is not distributed by this repository.
Download it from the
[`motus-robotics/pi0.5_robotwin2` Hugging Face repository](https://huggingface.co/motus-robotics/pi0.5_robotwin2)
and preserve the complete checkpoint directory, including its `assets/`
subdirectory, at:

```text
/home/ubuntu/Model/pi0.5_robotwin2
```

The published model card identifies it as a PyTorch-converted pi0.5
checkpoint fine-tuned in the RoboTwin 2.0 simulation environment.

### 2. Clone this repository

```bash
cd /home/ubuntu/Workspace
git clone git@github.com:z-R-rr/Event-triggered-replanning-for-VLA-control.git
cd Event-triggered-replanning-for-VLA-control
```

### 3. Inspect and apply the overlays

Set task-specific paths rather than modifying shell-wide variables:

```bash
ROBOTWIN_CHECKOUT=/home/ubuntu/Workspace/RoboTwin
OPENPI_CHECKOUT=/home/ubuntu/Workspace/openpi
```

Preview the exact files that will be replaced:

```bash
rsync -avni robotwin/ "${ROBOTWIN_CHECKOUT}/"
rsync -avni openpi/ "${OPENPI_CHECKOUT}/"
```

Apply the overlay only from clean or intentionally backed-up checkouts:

```bash
rsync -av robotwin/ "${ROBOTWIN_CHECKOUT}/"
rsync -av openpi/ "${OPENPI_CHECKOUT}/"
```

Then inspect both source trees:

```bash
git -C "${ROBOTWIN_CHECKOUT}" diff --stat
git -C "${OPENPI_CHECKOUT}" diff --stat
```

No additional Python package is introduced by this overlay. Use the existing
RoboTwin `.venv` and OpenPI environment created by their upstream installation
procedures.

## Running the pipeline

Replan-router input-feature selection 使用同一个版本化入口：

```bash
python3 robotwin/script/run_replan_router_feature_selection_pipeline.py \
  --stage validate
```

固定的 348-sample scene-grouped OOF、balanced Online-40 和 Result 1/2
输出协议见
[replan-router input-feature selection v2](docs/REPLAN_ROUTER_FEATURE_SELECTION_V2.md)。

Use the complete command sequence in the
[pipeline runbook](docs/PI05_REPLAN_PIPELINE_RUNBOOK.md). The stage order is:

1. inspect the task implementation and freeze a new result directory;
2. run the deterministic 600-episode six-r discovery grid;
3. audit fixed seeds, prompt plaintext, and fingerprints;
4. select r25-fail/other-r-success scenes;
5. generate mapped candidate windows and at most ten nodes per scene;
6. pass the 3+3 paired reproducibility smoke;
7. prepare and inspect the shared-control experiment plan;
8. run the resumable paired evaluation;
9. optionally run a separately reported second node round;
10. generate and audit `scene_outcomes`.

The intervention definition is exact:

```text
node t = execute actions 1..t,
         discard the remaining old chunk,
         infer again immediately before action t+1
```

### Forced-replan cadence modes and off-by-one semantics

The runtime supports two explicit cadence modes. The default remains the
original re-anchored behavior so existing experiments do not silently change.

#### Re-anchored cadence (default)

A forced replan starts a fresh full-length chunk and re-anchors subsequent
natural `r0` boundaries. For `r0=25` and candidate node `t=30`:

```text
initial inference -> execute actions 1..25
natural replan t=25 -> execute actions 26..30
forced replan t=30 -> discard the old chunk tail and infer before action 31
new chunk -> execute actions 31..55
natural replan t=55 -> execute actions 56..80
natural replan t=80 -> ...
```

The resulting non-initial replan nodes are:

```text
25, 30, 55, 80, ...
```

This is selected by the base scripts with
`pi05_absolute_r0_cadence=false`.

#### Absolute `r0` cadence (opt-in)

The forced replacement chunk is truncated at the next original absolute
`r0` boundary. Natural replans therefore remain anchored at
`r0, 2*r0, 3*r0, ...`. For the same `r0=25`, `t=30` intervention:

```text
initial inference -> execute actions 1..25
natural replan t=25 -> execute actions 26..30
forced replan t=30 -> discard the old chunk tail and infer before action 31
replacement chunk -> execute actions 31..50 (20 actions)
natural replan t=50 -> execute actions 51..75
natural replan t=75 -> ...
```

The resulting non-initial replan nodes are:

```text
25, 30, 50, 75, ...
```

Use the clearly named entry points:

```bash
python3 script/run_pi05_paired_replan_smoke_absolute_cadence.py [same smoke arguments]
python3 script/run_pi05_paired_replan_node_eval_absolute_cadence.py [same eval arguments]
```

They append `--absolute-r0-cadence`, which passes
`pi05_absolute_r0_cadence=true` to the policy. The generic smoke and evaluator
also accept that flag directly. Absolute cadence is rejected when dynamic-r is
enabled because the two execution schedules are not jointly defined.

The absolute-cadence smoke and evaluator additionally require every forced
replacement to execute exactly:

```text
r0 - (t mod r0)
```

actions and to terminate at the next absolute multiple of `r0`.

The node manifest and policy option use different indexing conventions:

```text
replan_after_actions = t
force_before_one_based_action = t + 1
```

For node `t=30`, the paired evaluator therefore records:

```json
{
  "replan_after_actions": 30,
  "force_before_one_based_action": 31
}
```

and passes:

```text
pi05_force_replan_before_actions=[31]
```

to the policy. Supplying the raw policy option as `[30]` would instead force a
replan before action 30, which is node `t=29`. Its re-anchored cadence would be
`25, 29, 54, 79, ...`; its absolute cadence would be
`25, 29, 50, 75, ...`.

### Merging candidate rounds for a cadence comparison

To evaluate exactly the union of first- and second-round nodes without
overwriting either source manifest:

```bash
python3 script/merge_pi05_replan_node_manifests.py \
  --input /path/to/primary_nodes.json \
  --input /path/to/round2_nodes.json \
  --output /new/path/to/union_nodes.json
```

The merger verifies a single `r0` per scene, deduplicates nodes, records
per-node source provenance, and refuses to overwrite an incompatible output.

### Offline replan-router feasibility experiment

`train_replan_router.py` does not start RoboTwin, call `sample_actions`, or
modify the pi0.5 inference pipeline. It reads completed absolute-cadence
shared-prefix pairs, excludes `keep=0/replan=0` pairs, freezes pi0.5, and
extracts one of two visual representations:

- `vision_encoder`: masked mean of each camera's projected SigLIP tokens,
  concatenated in high/left-wrist/right-wrist order;
- `vlm_hidden`: masked mean of final contextual PaliGemma hidden states at
  valid language-token positions.

The router has two binary outcome heads:

```text
p_keep   = P(keep succeeds | zv, za)
p_replan = P(replan succeeds | zv, za)
replan iff p_replan - p_keep > lambda
```

`router_input=vision_action` adds a small MLP encoder for the archived
`[H, action_dim]` old action chunk. `router_input=vision` is the visual-only
ablation. Both configurations remain below one million trainable parameters.
The deterministic 70/15/15 split is grouped by scene seed, so candidate nodes
from one scene cannot cross splits.

Run from an environment where `python` is the OpenPI interpreter:

```bash
cd /home/ubuntu/Workspace/Event-triggered-replanning-for-VLA-control/robotwin/script

python train_replan_router.py \
  --feature_type vision_encoder \
  --router_input vision_action \
  --task move_playingcard_away
```

For another task such as `pick_dual_bottles`, pass its absolute-cadence paired
roots explicitly when they are not discoverable under the standard result
locations:

```bash
python train_replan_router.py \
  --feature_type vlm_hidden \
  --router_input vision \
  --task pick_dual_bottles \
  --data-root /absolute/path/to/paired_eval
```

The script writes an immutable feature NPZ and manifest, a router checkpoint,
and an evaluation JSON. Existing compatible features are reused; router runs
are never overwritten.

### Offline router feature ablation

The same script also runs the fixed-protocol V0--V7 input ablation while
reusing the exact dataset fingerprint and scene-grouped split:

```bash
python train_replan_router.py \
  --task move_playingcard_away \
  --feature_config V5

python train_replan_router.py \
  --task move_playingcard_away \
  --run_feature_ablation
```

The configurations add chunk state, the unexecuted action tail, future-action
statistics, and/or `z_visual(t)-z_visual(t-5)` to the frozen vision feature.
Temporal configurations strictly skip samples without an archived same-scene
observation at exactly `t-5`; they retain the original scene assignment and
record filtered counts per split. The action-tail MLP never sees the executed
chunk prefix. A separate `C3` diagnostic exposes the legacy full-chunk input
without adding it to the prescribed V0--V7 summary.

The optional `E1` geometry-aware action diagnostic is also outside V0--V7:

```bash
python train_replan_router.py \
  --task move_playingcard_away \
  --feature_config E1
```

It differences future qpos targets from the decision observation state,
integrates the deltas, runs offline FK with RoboTwin's ALOHA URDF, and samples
five dual-EEF pose waypoints plus their Cartesian/angular/gripper velocity
profile. The resulting 190-D descriptor is encoded by the same `128 -> 64`
action MLP. This path uses only an offline fixed-root articulation for FK; it
starts no RoboTwin task, sends no control action, and changes no online code.

Outputs are isolated under
`temp/outputs/replan_router_feature_ablation/{configs,checkpoints,evaluations,logs}`.
Each evaluation contains test metrics and threshold sweeps at
`0.05, 0.1, 0.2, 0.3, 0.5`; the root also contains
`evaluation_feature_ablation.json` and `feature_ablation_summary.md`.

## Read-only acceptance audit

After the pipeline completes:

```bash
cd /home/ubuntu/Workspace/RoboTwin

python3 script/audit_pi05_replan_pipeline.py \
  --root /absolute/path/to/the/task/run
```

Pass every paired round explicitly when a run has multiple rounds:

```bash
python3 script/audit_pi05_replan_pipeline.py \
  --root "$RUN_ROOT" \
  --paired-summary "$RUN_ROOT/paired_replan_node_eval_r25_deterministic/paired_summary.json" \
  --paired-summary "$RUN_ROOT/paired_replan_node_eval_r25_round2_no_rescue_5each_deterministic/paired_summary.json"
```

The audit launches no GPU process and writes nothing into the result
directory. It checks all six trace sets, ordered seeds, prompt plaintext and
fingerprints, deterministic server logs, candidate caps, paired validity, and
the final scene-level success count.

## Validated minimal experiment

For `move_playingcard_away`, using the default re-anchored cadence:

- discovery: 600 episodes;
- r25 baseline: 73/100;
- contrast scenes: 22;
- primary candidate nodes: 129;
- reproducibility smoke: 6 episodes;
- primary paired evaluation: 22 controls + 129 forced;
- optional second round: 11 controls + 55 forced;
- distinct rescued scenes across both rounds: 14;
- final r25 plus one-replan outcome: 87/100.

Node-level rescue counts must not be added directly to the scene-level success
rate. A scene with several successful rescue nodes contributes only one final
success.

### Validated absolute-cadence comparison

The opt-in `25/30/50/75` implementation was separately validated on the union
of both candidate rounds:

- merged manifest: 22 scenes, 184 unique nodes (`129 + 55`, no overlap);
- strict reproducibility smoke: 3 controls + 3 forced, all gates passed;
- formal evaluation: 22 shared controls + 184 forced treatments;
- total new execution including smoke: 212 episodes;
- valid causal pairs: 184/184;
- node-level rescues: 56/184 (30.43%);
- scenes with at least one rescue node: 16/22;
- oracle scene-level result: `73 + 16 = 89/100`.

The 89% value assumes choosing a node already shown to rescue each scene; it
is not the accuracy of a label-free online trigger. Cadence changes the entire
post-intervention trajectory: relative to the two default-cadence rounds, the
absolute mode gained six rescued scenes and lost four rather than being a
strict superset.

## Reproducibility and scope

This is an exploratory, label-assisted causal evaluation pipeline, not a
label-free online event trigger. Candidate scenes and nodes use discovery-grid
outcomes. Final task success is used only to validate paired interventions.

Do not:

- switch deterministic mode between discovery and paired evaluation;
- substitute unstable seeds or prompt plaintext;
- use forced-only runs as causal evidence;
- mix second-round adaptive nodes into the preregistered primary result;
- overwrite an existing result directory.

## Licenses

The overlay contains files derived from both RoboTwin and OpenPI. Their
respective license texts are preserved under [`LICENSES/`](LICENSES/).
