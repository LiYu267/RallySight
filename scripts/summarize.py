"""Audit the RallySight showcase traces with the Python standard library."""
import argparse
from bisect import bisect_right
import json
from pathlib import Path
from statistics import fmean


def longest_streak(contacts):
    longest = length = 0
    previous = None
    for contact in contacts:
        if contact["phase"] != "strike" or not contact["forward"]:
            length, previous = 0, None
            continue
        side = contact["side"]
        length = 1 if side == previous else length + 1
        longest = max(longest, length)
        previous = side
    return longest


def summarize(trace):
    contacts = trace["contacts"]
    frames = trace["trajectory"]
    updates = trace["anticipation"]
    times = [contact["t_s"] for contact in contacts]
    speeds = []
    for a, b in zip(frames, frames[1:]):
        index = bisect_right(times, a["t_s"]) - 1
        if index < 0 or not a["idle"]["left"] or not b["idle"]["left"]:
            continue
        last = contacts[index]
        if last["side"] == "left" and last["forward"] and last["phase"] == "strike" and a["t_s"] - last["t_s"] <= 1.6:
            speeds.append(abs(b["left_xy"][1] - a["left_xy"][1]) / (b["t_s"] - a["t_s"]))
    steps = [
        abs(b["ready_y_m"] - a["ready_y_m"])
        for a, b in zip(updates, updates[1:])
        if b["t_s"] - a["t_s"] < 0.051
    ]
    return {
        "seed": trace["seed"],
        "method": trace["method"],
        "returns": trace["alternating_returns"],
        "longest_active_streak": longest_streak(contacts),
        "mean_abs_lateral_speed_m_s": round(fmean(speeds), 3) if speeds else None,
        "max_adjacent_anticipation_target_step_cm": round(max(steps) * 100, 2) if steps else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=Path(__file__).resolve().parents[1] / "evidence")
    args = parser.parse_args()
    traces = [json.loads(file.read_text()) for file in sorted(args.evidence.glob("41*.json"))]
    if not traces:
        parser.error("No evaluation traces found")
    print(json.dumps([summarize(trace) for trace in traces], indent=2))


if __name__ == "__main__":
    main()
