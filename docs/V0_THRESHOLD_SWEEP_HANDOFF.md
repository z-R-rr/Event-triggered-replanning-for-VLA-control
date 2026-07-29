# V0 threshold sweep handoff

Last updated: 2026-07-27 UTC

## 1. Active run

The reset-safe correction run is active:

```text
runner PID: 4064479
session ID: 4064479
parent PID: 1
mode:       setsid + nohup, independent of Codex/PTTY
plan:       55 episodes
```

Check progress:

```bash
cd /home/ubuntu/Workspace/Event-triggered-replanning-for-VLA-control
sed -n '1,120p' \
  temp/outputs/move_playingcard_away_v0_threshold_sweep_r25_failures_corrected/progress.json
ps -p 4064479 -o pid,ppid,sid,stat,etime,cmd
```

At handoff time:

```text
completed: 14/55
control:    7/27
lambda 0.3: 1/3
lambda 0.5: 6/25
```

Runner output:

```text
temp/outputs/move_playingcard_away_v0_threshold_sweep_r25_failures_corrected/
  logs/correction_runner.log
```

Do not stop or relaunch the run while PID `4064479` is alive.

## 2. Question being answered

Use the 27 scenes that failed in the recorded deterministic
`move_playingcard_away`, `H=50`, `r0=25` run beginning at seed `100000`.
Keep the trained V0 router fixed and compare:

```text
lambda = 0.05, 0.1, 0.2, 0.3, 0.5
query interval = 5
max router replans = 1
absolute natural r0 cadence = 25
```

V0 input is frozen pi0.5 vision only. The decision is:

```text
p_keep   = sigmoid(keep_success_logit)
p_replan = sigmoid(replan_success_logit)
trigger iff p_replan - p_keep > lambda
```

Source failure manifest:

```text
temp/outputs/move_playingcard_away_v0_threshold_sweep_r25_failures/
  inputs/r25_failure_seed_manifest.json
```

The 27 seeds are:

```text
100002, 100005, 100006, 100017, 100025, 100026, 100033,
100034, 100039, 100044, 100045, 100046, 100049, 100050,
100053, 100055, 100061, 100065, 100070, 100072, 100073,
100074, 100076, 100078, 100082, 100094, 100099
```

## 3. Original sweep and validity failure

The original sweep completed all 135 episodes:

| Lambda | Raw success | Triggered | Median trigger |
|---:|---:|---:|---:|
| 0.05 | 8/27 | 27/27 | 15 |
| 0.10 | 8/27 | 27/27 | 15 |
| 0.20 | 7/27 | 27/27 | 20 |
| 0.30 | 6/27 | 27/27 | 20 |
| 0.50 | 14/27 | 19/27 | 55 |

Do not use the raw `lambda=0.5` result. The dynamic worker queue could run the
same scene twice in succession on one persistent policy server. All thresholds
used the same `(episode_seed, episode_id)`, so `EpisodeSeededPolicy` did not
reset `_primary_inference_index` for the second replay.

Observed contamination:

```text
lambda 0.05:  0/27
lambda 0.10:  0/27
lambda 0.20:  0/27
lambda 0.30:  3/27  (100006, 100033, 100078)
lambda 0.50: 25/27
```

The failure was visible before any router decision:

```text
lambda 0.5 had a different first-query observation in 25/27 scenes
lambda 0.3 had a different first-query observation in 3/27 scenes
```

Original artifacts remain read-only:

```text
temp/outputs/move_playingcard_away_v0_threshold_sweep_r25_failures/
  experiment_plan.json
  progress.json
  v0_threshold_sweep_summary.json
```

## 4. Correction design

The base runner now assigns a unique reset-only occurrence token to every
`(threshold, scene)`:

```text
occurrence_token = threshold_index * 100000 + source_failure_index
```

`episode_id` forces a server reset but does not contribute sampling entropy;
`episode_seed` remains the only episode-specific entropy source.

The correction run adds only 55 episodes:

| Role | Episodes |
|---|---:|
| contemporaneous control | 27 |
| corrected lambda 0.3 | 3 |
| corrected lambda 0.5 | 25 |

It reuses only prefix-clean original treatments:

```text
lambda 0.05: all 27
lambda 0.10: all 27
lambda 0.20: all 27
lambda 0.30: 24 clean + 3 corrected
lambda 0.50:  2 clean + 25 corrected
```

Every final pair must pass:

1. exact pre-trigger observation fingerprint match with current control;
2. exact pre-trigger action-target match;
3. at most one trigger at an allowed query node;
4. replacement uses the trigger observation and absolute r0 cadence;
5. no-trigger treatment has the same targets and outcome as control.

Only `valid_pairs` may be used for conclusions.

## 5. Implementation and outputs

Key files:

| File | Purpose |
|---|---|
| `robotwin/script/run_pi05_online_v0_threshold_sweep.py` | Full threshold sweep; now reset-safe |
| `robotwin/script/run_pi05_online_v0_threshold_sweep_correction.py` | Minimal 55-episode correction, merge, and strict audit |
| `openpi/scripts/serve_robotwin_policy.py` | Episode-seeded action sampling and inference-index reset |
| `openpi/scripts/serve_robotwin_router_policy.py` | Frozen V0 scoring and threshold decision |
| `robotwin/policy/pi05_remote.py` | Online query clocks, trigger, and max-replan enforcement |

Correction output root:

```text
temp/outputs/move_playingcard_away_v0_threshold_sweep_r25_failures_corrected/
```

Expected final summary:

```text
v0_threshold_sweep_corrected_summary.json
```

The summary contains, per threshold:

```text
valid_pairs
control_successes
treatment_successes
triggered_pairs
rescues
harms
both_success
both_failure
```

## 6. Completion procedure

When `completed_episodes == 55`:

1. Require PID `4064479` and ports `8400-8403` to exit automatically.
2. Require `v0_threshold_sweep_corrected_summary.json` to exist.
3. Require `valid_pairs == 27` for every threshold.
4. Compare thresholds by net gain: `rescues - harms`.
5. Report trigger rate and median trigger node with success outcomes.

Print the compact result:

```bash
cd /home/ubuntu/Workspace/Event-triggered-replanning-for-VLA-control
/home/ubuntu/Workspace/openpi/.venv/bin/python - <<'PY'
import json
from pathlib import Path

path = Path(
    "temp/outputs/"
    "move_playingcard_away_v0_threshold_sweep_r25_failures_corrected/"
    "v0_threshold_sweep_corrected_summary.json"
)
data = json.loads(path.read_text())
for name, row in data["threshold_summaries"].items():
    print(
        name,
        "valid=", row["valid_pairs"],
        "control=", row["control_successes"],
        "router=", row["treatment_successes"],
        "triggered=", row["triggered_pairs"],
        "rescues=", row["rescues"],
        "harms=", row["harms"],
    )
PY
```

If the runner is absent while progress is incomplete, first verify that ports
`8400-8403` have no owner. The launcher is resume-safe because completed cases
with `metrics.json` are skipped:

```bash
cd /home/ubuntu/Workspace/Event-triggered-replanning-for-VLA-control
setsid nohup /home/ubuntu/Workspace/openpi/.venv/bin/python \
  robotwin/script/run_pi05_online_v0_threshold_sweep_correction.py \
  --output-dir \
  /home/ubuntu/Workspace/Event-triggered-replanning-for-VLA-control/temp/outputs/move_playingcard_away_v0_threshold_sweep_r25_failures_corrected \
  > temp/outputs/move_playingcard_away_v0_threshold_sweep_r25_failures_corrected/logs/correction_runner.log \
  2>&1 < /dev/null &
```
