# Capability-Aware Multi-Robot Sim

MAE 577 final project

Proof that multi-robot coordination algorithms produce **measurably
different behavior** when they treat user physical capability (reach, mobility,
field of view) as a first-class variable in task allocation.

## Quick start

```bash
cd capability_aware_sim
pip install pygame numpy matplotlib   # one-time
python main.py
```

## Reproducing the headline numbers

The centralized allocators (baseline, adaptive) are fully deterministic — seed
has no effect on their output.  To reproduce the paper's primary condition:

```bash
# Headless single run — prints metrics to stdout, writes no files
python -c "
from sim_runner import run_headless
s, _ = run_headless('baseline', reach_right=0.8, reach_left=0.25, seed=42, profile_name='baseline_test')
print(s)
s, _ = run_headless('adaptive', reach_right=0.8, reach_left=0.25, seed=42, profile_name='baseline_test')
print(s)
s, _ = run_headless('stigmergic', reach_right=0.8, reach_left=0.25, seed=42, profile_name='baseline_test')
print(s)
"
```

Expected output (baseline_test profile, seed=42):

| Allocator  | Total time | Repositions | Unreachable |
|------------|-----------|-------------|-------------|
| Baseline   | 16.85 s   | 1           | 1           |
| Adaptive   | 18.05 s   | 0           | 0           |
| Stigmergic | 36.10 s   | 0           | 0           |

> Note: live pygame sim times (16.6 s / 17.8 s) differ slightly from headless
> due to 60 Hz clock jitter vs fixed dt=0.05 s stepping.  All sweep and paper
> figures use the headless runner for strict reproducibility.

## Reproducing the capability-profile sweep

```bash
python sweep.py --output sweep_results.csv
```

This runs **180 conditions** (6 profiles × 3 allocators × 10 seeds) in ~10 s.
Output:
- `sweep_results.csv` — one row per run, all metrics
- `runs/` — one JSON per run with per-task breakdown

For a quick sanity check (6 runs only):

```bash
python sweep.py --smoke
```

## Regenerating the figures

```bash
python plot_results.py sweep_results.csv
```

Writes:
- `summary.png` — three-way bar chart (Baseline | Adaptive | Stigmergic) for
  the `baseline_test` profile; stigmergic bar shows mean ± min/max across seeds
- `sweep_chart.png` — 2×2 line chart, x-axis = reach asymmetry, one line per
  allocator; this is the paper's Section V figure

Custom options:

```bash
python plot_results.py sweep_results.csv --summary-out my_summary.png --profile severe
python plot_results.py sweep_results.csv --sweep-only
```

## Controls (live pygame sim)

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `S` | Save screenshot → `screenshot.png` |
| `F` | Toggle 3× fast-forward |

Run modes:

```bash
python main.py                        # side-by-side: Baseline vs Adaptive
python main.py --allocator=stigmergic # three panels: Baseline | Adaptive | Stigmergic
python main.py --mode=baseline        # single panel, full size
python main.py --mode=adaptive
python main.py --mode=stigmergic
```

## File overview

| File | Role |
|------|------|
| `world.py` | Room dimensions (5 m × 4 m), item positions, handoff base |
| `agents.py` | `RobotAgent` (straight-line + soft avoidance), `UserAgent` (asymmetric reach, FoV) |
| `tasks.py` | `Task` dataclass + 5-task scripted list |
| `allocators.py` | `baseline_allocator` (travel cost only), `adaptive_allocator` (weighted travel + handoff penalty) |
| `stigmergic.py` | `StigmergicRobot` (local sensing + seeded exploration), `StigmergicPanelState` |
| `metrics.py` | `MetricsLog`, `plot_summary` (two-panel chart for live sim) |
| `sim_runner.py` | **Headless engine** — no pygame; used by sweep.py and for reproducibility |
| `sweep.py` | Capability-profile sweep runner; writes `sweep_results.csv` + per-task JSONs |
| `plot_results.py` | `summary.png` (three-way bar) + `sweep_chart.png` (line chart over asymmetry) |
| `main.py` | Pygame loop, panels, sidebar, live chart |

## Metric definitions

### Core metrics
| Metric | Meaning |
|--------|---------|
| `total_time` | Makespan — sim seconds to last delivery |
| `repositions` | Times user moved because handoff was unreachable |
| `unreachable` | Deliveries placed outside user's reachable workspace |
| `deliveries` | Should be 5 for all conditions |

### HRC fluency (Hoffman 2019, sampled at 0.1 s intervals)
| Metric | Meaning |
|--------|---------|
| `human_idle_time` | Fraction of run time user is not engaged |
| `robot_idle_time` | Fraction of run time all robots are idle |
| `concurrent_activity` | Fraction where both human and robots are active |
| `functional_delay` | Fraction where both are idle simultaneously |

### Capability-aware metrics
| Metric | Meaning |
|--------|---------|
| `strong_side_fraction` | Fraction of handoffs on user's dominant (right) side |
| `mean_handoff_excess` | Average distance handoffs fell outside reachable workspace |
| `max_handoff_excess` | Worst single handoff excess |

### Efficiency metrics
| Metric | Meaning |
|--------|---------|
| `total_robot_distance` | Sum of all robot trajectories (m) |
| `mean_task_duration` | Mean per-task completion time |
| `task_completion_variance` | Variance in per-task completion times |

## Seed list

Ten seeds for ten trials per condition:

```python
SEEDS = [42, 137, 271, 314, 628, 1024, 1729, 2718, 3141, 6283]
```

Centralized allocators (baseline, adaptive) produce identical results across
all seeds.  Stigmergic exploration is seeded via `numpy.random.default_rng` so
same seed → identical run; different seed → different exploration path.

## Capability profiles

| Profile | reach_right | reach_left | asymmetry |
|---------|-------------|------------|-----------|
| symmetric | 0.5 | 0.5 | 0.0 |
| mild_asymmetric | 0.6 | 0.4 | 0.2 |
| moderate | 0.7 | 0.3 | 0.4 |
| baseline_test | 0.8 | 0.25 | 0.55 |
| severe | 0.9 | 0.15 | 0.75 |
| extreme | 1.0 | 0.10 | 0.90 |

## Tunable constants (not to be changed for paper)

| Constant | File | Default | Meaning |
|----------|------|---------|---------|
| `REACH_RIGHT` | `agents.py` | 0.8 m | User dominant-side reach (live sim) |
| `REACH_LEFT` | `agents.py` | 0.25 m | User off-side reach (live sim) |
| `ROBOT_MAX_SPEED` | `agents.py` | 0.6 m/s | Robot top speed |
| `W_TRAVEL` | `allocators.py` | 1.0 | Weight on robot travel cost |
| `W_HANDOFF` | `allocators.py` | 2.5 | Weight on handoff reachability penalty |
| `SENSE_RADIUS` | `stigmergic.py` | 2.0 m | Swarm item/robot sensing range |
| `BEACON_RADIUS` | `stigmergic.py` | 1.5 m | Range at which robot learns user reach |
