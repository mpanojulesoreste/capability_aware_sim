"""Capability-profile sweep: 6 profiles × 3 allocators × 10 seeds."""

import argparse
import csv
import json
import os
import sys
import time

from sim_runner import run_headless

SWEEP_PROFILES = [
    ("symmetric",       0.5,  0.5),
    ("mild_asymmetric", 0.6,  0.4),
    ("moderate",        0.7,  0.3),
    ("baseline_test",   0.8,  0.25),
    ("severe",          0.9,  0.15),
    ("extreme",         1.0,  0.10),
]

ALLOCATORS = ["baseline", "adaptive", "stigmergic"]

SEEDS = [42, 137, 271, 314, 628, 1024, 1729, 2718, 3141, 6283]

CSV_COLUMNS = [
    "profile_name", "reach_right", "reach_left", "allocator", "seed",
    "total_time", "repositions", "unreachable", "deliveries",
    "strong_side_fraction", "mean_handoff_excess",
    "total_robot_distance",
    "h_idle", "r_idle", "c_act", "f_del",
]


def _run_one(profile_name, reach_right, reach_left, allocator, seed, runs_dir):
    """Run one condition, write per-task JSON, return CSV row dict."""
    summary, records = run_headless(
        allocator_name=allocator,
        reach_right=reach_right,
        reach_left=reach_left,
        seed=seed,
        profile_name=profile_name,
    )

    json_path = os.path.join(runs_dir, f"{profile_name}_{allocator}_{seed}.json")
    with open(json_path, "w") as f:
        json.dump({
            "profile":    profile_name,
            "reach_right": reach_right,
            "reach_left":  reach_left,
            "allocator":  allocator,
            "seed":       seed,
            "summary":    summary,
            "tasks":      records,
        }, f, indent=2)

    return {
        "profile_name":         profile_name,
        "reach_right":          reach_right,
        "reach_left":           reach_left,
        "allocator":            allocator,
        "seed":                 seed,
        "total_time":           summary["total_time"],
        "repositions":          summary["repositions"],
        "unreachable":          summary["unreachable"],
        "deliveries":           summary["deliveries"],
        "strong_side_fraction": summary["strong_side_fraction"],
        "mean_handoff_excess":  summary["mean_handoff_excess"],
        "total_robot_distance": summary["total_robot_distance"],
        "h_idle":               summary["human_idle_time"],
        "r_idle":               summary["robot_idle_time"],
        "c_act":                summary["concurrent_activity"],
        "f_del":                summary["functional_delay"],
    }


def run_sweep(profiles, allocators, seeds, output_csv, runs_dir, verbose=True):
    os.makedirs(runs_dir, exist_ok=True)

    total = len(profiles) * len(allocators) * len(seeds)
    done  = 0
    rows  = []
    t0    = time.time()

    for profile_name, reach_right, reach_left in profiles:
        for allocator in allocators:
            for seed in seeds:
                row = _run_one(profile_name, reach_right, reach_left,
                               allocator, seed, runs_dir)
                rows.append(row)
                done += 1
                if verbose:
                    elapsed = time.time() - t0
                    eta = elapsed / done * (total - done)
                    print(f"  [{done:3d}/{total}] {profile_name:16s} {allocator:12s} "
                          f"seed={seed:5d}  t={row['total_time']:6.2f}s "
                          f"repos={row['repositions']}  "
                          f"ETA {eta:5.0f}s", flush=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    if verbose:
        print(f"\nSweep complete — {len(rows)} runs in {time.time()-t0:.1f}s")
        print(f"CSV  → {output_csv}")
        print(f"JSONs → {runs_dir}/")

    return rows


def main():
    parser = argparse.ArgumentParser(description="Capability-profile sweep")
    parser.add_argument("--output",  default="sweep_results.csv",
                        help="Output CSV path")
    parser.add_argument("--runs-dir", default="runs",
                        help="Directory for per-task JSON files")
    parser.add_argument("--smoke",   action="store_true",
                        help="Quick smoke test: 1 profile × 3 allocators × 2 seeds")
    parser.add_argument("--profile", default=None,
                        help="Run only this profile name")
    args = parser.parse_args()

    profiles   = SWEEP_PROFILES
    allocators = ALLOCATORS
    seeds      = SEEDS

    if args.smoke:
        profiles   = [("baseline_test", 0.8, 0.25)]
        seeds      = [42, 137]
        print("Smoke test: 1 profile × 3 allocators × 2 seeds = 6 runs")

    elif args.profile:
        matches = [p for p in SWEEP_PROFILES if p[0] == args.profile]
        if not matches:
            print(f"Unknown profile '{args.profile}'. Choose from:")
            for p in SWEEP_PROFILES:
                print(f"  {p[0]}")
            sys.exit(1)
        profiles = matches

    print(f"Running {len(profiles)} profile(s) × {len(allocators)} allocators "
          f"× {len(seeds)} seeds = {len(profiles)*len(allocators)*len(seeds)} runs")

    run_sweep(profiles, allocators, seeds, args.output, args.runs_dir)


if __name__ == "__main__":
    main()
