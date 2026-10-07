"""
Compute duration statistics (max, percentiles, histogram) over a QA-sample
dataset in the same format consumed by the VideoChat-Flash converter script.

Input format (dict keyed by qid), each value a list:
[
    {"role": "system", ...},
    {"role": "user", ...},
    {"role": "assistant", ...},
    {"start_time": <float seconds>, "end_time": <float seconds>, "procedure_type": <str>},
]

Usage:
    python analyze_durations.py /path/to/input.json
    python analyze_durations.py /path/to/input.json --fps 1   # also estimate frame counts
"""

import argparse
import json
import statistics


def get_duration(sample_messages: list):
    """Extract (start_time, end_time, duration) from one sample's meta block."""
    try:
        meta = sample_messages[3]
        start_time = float(meta.get("start_time", 0))
        end_time = float(meta.get("end_time", 0))
        return start_time, end_time, end_time - start_time
    except (IndexError, KeyError, TypeError, ValueError):
        return None


def percentile(sorted_vals, pct):
    """Simple percentile on an already-sorted list (0-100)."""
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * (pct / 100)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def print_histogram(durations, num_bins=20):
    lo, hi = min(durations), max(durations)
    if lo == hi:
        print(f"All {len(durations)} durations are {lo:.1f}s — no spread to histogram.")
        return
    width = (hi - lo) / num_bins
    bins = [0] * num_bins
    for d in durations:
        idx = min(int((d - lo) / width), num_bins - 1)
        bins[idx] += 1
    max_count = max(bins)
    print(f"\nHistogram ({num_bins} bins, {lo:.1f}s to {hi:.1f}s):")
    for i, count in enumerate(bins):
        bin_lo = lo + i * width
        bin_hi = bin_lo + width
        bar = "#" * int(50 * count / max_count) if max_count else ""
        print(f"  [{bin_lo:7.1f}s - {bin_hi:7.1f}s) {count:6d} | {bar}")


def main():
    parser = argparse.ArgumentParser(description="Compute duration stats over a QA dataset JSON.")
    parser.add_argument("--input", default="/iopsstor/scratch/cscs/lpanta32/focus/lapchole_train_overlay_long.json",
    help="Path to input JSON file (dict keyed by qid).")
    parser.add_argument(
        "--fps",
        type=float,
        default=1,
        help="If given, also estimate sampled-frame counts at this fps "
             "(clamped to --lowbound/--upbound), matching dynamic_fps1-style sampling.",
    )
    parser.add_argument("--lowbound", type=int, default=32, help="frames_lowbound (default 32)")
    parser.add_argument("--upbound", type=int, default=800, help="frames_upbound (default 800)")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    durations = []
    skipped = 0
    for qid, sample_messages in data.items():
        result = get_duration(sample_messages)
        if result is None:
            skipped += 1
            continue
        start_time, end_time, duration = result
        if duration <= 0:
            skipped += 1
            continue
        durations.append(duration)

    if not durations:
        print("No valid durations found — check the input format.")
        return

    durations.sort()
    n = len(durations)

    print(f"Total samples: {len(data)}")
    print(f"Valid samples: {n}  (skipped/malformed: {skipped})")
    print(f"Min duration:    {durations[0]:8.1f}s  ({durations[0]/60:.2f} min)")
    print(f"Max duration:    {durations[-1]:8.1f}s  ({durations[-1]/60:.2f} min)")
    print(f"Mean duration:   {statistics.mean(durations):8.1f}s")
    print(f"Median duration: {statistics.median(durations):8.1f}s")
    print(f"Stdev duration:  {statistics.pstdev(durations):8.1f}s")
    print()
    for p in [50, 75, 90, 95, 99, 99.5, 100]:
        val = percentile(durations, p)
        print(f"  p{p:<5}: {val:8.1f}s  ({val/60:.2f} min)")

    print_histogram(durations)

    if args.fps is not None:
        print(f"\n--- Estimated sampled frame counts at fps={args.fps}, "
              f"clamped to [{args.lowbound}, {args.upbound}] ---")
        frame_counts = sorted(
            min(max(int(round(d * args.fps)), args.lowbound), args.upbound) for d in durations
        )
        at_cap = sum(1 for fc in frame_counts if fc >= args.upbound)
        print(f"Samples hitting the upbound ({args.upbound} frames): "
              f"{at_cap} / {n} ({100*at_cap/n:.1f}%)")
        for p in [50, 75, 90, 95, 99, 100]:
            val = percentile(frame_counts, p)
            print(f"  p{p:<5} frames: {val:.0f}")


if __name__ == "__main__":
    main()