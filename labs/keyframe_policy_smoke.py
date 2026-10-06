#!/usr/bin/env python3
"""Deterministic multi-signal keyframe policy with hysteresis and ablations."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
DT = 0.1
FRAMES = 360
MIN_SPACING_FRAMES = 12


@dataclass
class FrameSignals:
    index: int
    time_s: float
    segment: str
    position_m: np.ndarray
    yaw_deg: float
    texture: float
    compute_load: float


def make_sequence() -> list[FrameSignals]:
    rng = np.random.default_rng(61)
    position = np.zeros(3)
    yaw = 0.0
    result: list[FrameSignals] = []
    for index in range(FRAMES):
        if index < 70:
            segment, step, yaw_step = "lateral_a", np.array([0.006, 0.0, 0.0]), 0.0
        elif index < 130:
            segment, step, yaw_step = "pure_rotation", np.zeros(3), 1.0
        elif index < 210:
            segment, step, yaw_step = "forward_low_parallax", np.array([0.0, 0.0, 0.008]), 0.0
        elif index < 290:
            segment, step, yaw_step = "lateral_b", np.array([0.0, 0.006, 0.0]), 0.0
        elif index < 320:
            segment, step, yaw_step = "weak_texture_stop", np.zeros(3), 0.0
        else:
            segment, step, yaw_step = "lateral_recovery", np.array([0.006, 0.0, 0.0]), 0.0
        if index:
            position = position + step
            yaw += yaw_step
        texture = 0.34 if segment == "weak_texture_stop" else 1.0
        texture *= float(np.clip(1.0 + rng.normal(0.0, 0.018), 0.25, 1.0))
        compute_load = 0.93 if 235 <= index < 253 else 0.42 + 0.04 * np.sin(index * 0.11)
        result.append(FrameSignals(index, index * DT, segment, position.copy(), yaw, texture, float(compute_load)))
    return result


def relative_metrics(frame: FrameSignals, anchor: FrameSignals) -> dict[str, float | int]:
    displacement = frame.position_m - anchor.position_m
    translation = float(np.linalg.norm(displacement))
    lateral = float(np.linalg.norm(displacement[:2]))
    rotation = abs(frame.yaw_deg - anchor.yaw_deg)
    parallax = float(np.degrees(np.arctan2(lateral, 6.0)))
    overlap = float(np.clip(np.exp(-translation / 0.55 - rotation / 50.0), 0.0, 1.0))
    deterministic_jitter = 4.0 * np.sin(frame.index * 0.37)
    tracked = int(np.clip(240.0 * overlap * frame.texture + deterministic_jitter, 0.0, 260.0))
    return {
        "translation_m": translation,
        "lateral_m": lateral,
        "rotation_deg": rotation,
        "parallax_deg": parallax,
        "overlap": overlap,
        "tracked_features": tracked,
        "elapsed_s": frame.time_s - anchor.time_s,
    }


def run_full_policy(sequence: list[FrameSignals]) -> tuple[list[dict], list[int]]:
    keyframes = [0]
    decisions: list[dict] = []
    armed_count = 0
    for frame in sequence:
        anchor = sequence[keyframes[-1]]
        metrics = relative_metrics(frame, anchor)
        geometry_ready = metrics["lateral_m"] >= 0.10 and metrics["parallax_deg"] >= 0.90
        view_change = metrics["rotation_deg"] >= 25.0 and metrics["overlap"] <= 0.70
        track_pressure = (
            90 <= metrics["tracked_features"] < 125
            and (metrics["parallax_deg"] >= 0.55 or metrics["rotation_deg"] >= 18.0)
        )
        timeout_ready = metrics["elapsed_s"] >= 3.0 and (
            metrics["parallax_deg"] >= 0.70 or metrics["rotation_deg"] >= 20.0
        )
        triggers = []
        if geometry_ready:
            triggers.append("geometry")
        if view_change:
            triggers.append("view_change")
        if track_pressure:
            triggers.append("track_pressure")
        if timeout_ready:
            triggers.append("timeout_with_information")
        candidate = bool(triggers)
        spacing_ok = frame.index - keyframes[-1] >= MIN_SPACING_FRAMES
        compute_ok = frame.compute_load < 0.80
        tracking_ok = metrics["tracked_features"] >= 90
        guards_ok = spacing_ok and compute_ok and tracking_ok
        armed_count = armed_count + 1 if candidate and guards_ok else 0
        insert = frame.index != 0 and armed_count >= 2
        if frame.index == 0:
            reason = "bootstrap"
            inserted = 1
        elif insert:
            reason = "+".join(triggers)
            inserted = 1
            keyframes.append(frame.index)
            armed_count = 0
        else:
            inserted = 0
            if candidate and not spacing_ok:
                reason = "blocked_min_spacing"
            elif candidate and not compute_ok:
                reason = "blocked_compute_budget"
            elif candidate and not tracking_ok and frame.segment == "weak_texture_stop":
                reason = "defer_weak_texture"
            elif candidate and not tracking_ok:
                reason = "blocked_tracking_floor"
            elif candidate and guards_ok:
                reason = "hysteresis_arming"
            elif frame.segment == "weak_texture_stop" and metrics["tracked_features"] < 90:
                reason = "defer_weak_texture"
            elif frame.segment == "forward_low_parallax" and metrics["translation_m"] >= 0.10:
                reason = "defer_low_parallax"
            else:
                reason = "hold"
        decisions.append(
            {
                "frame": frame.index,
                "time_s": frame.time_s,
                "segment": frame.segment,
                **metrics,
                "compute_load": frame.compute_load,
                "candidate": int(candidate),
                "spacing_ok": int(spacing_ok),
                "compute_ok": int(compute_ok),
                "tracking_ok": int(tracking_ok),
                "hysteresis_count": armed_count,
                "inserted": inserted,
                "reason": reason,
                "anchor_keyframe": keyframes[-2] if inserted and frame.index != 0 else keyframes[-1],
            }
        )
    return decisions, keyframes


def run_ablation(sequence: list[FrameSignals], mode: str) -> list[int]:
    keyframes = [0]
    for frame in sequence[1:]:
        metrics = relative_metrics(frame, sequence[keyframes[-1]])
        spacing_ok = frame.index - keyframes[-1] >= MIN_SPACING_FRAMES
        if mode == "rotation_only":
            insert = metrics["rotation_deg"] >= 8.0
        elif mode == "translation_only":
            insert = metrics["translation_m"] >= 0.10
        elif mode == "timer_only":
            insert = metrics["elapsed_s"] >= 1.5
        else:
            raise ValueError(mode)
        if spacing_ok and insert:
            keyframes.append(frame.index)
    return keyframes


def summarize(name: str, keyframes: list[int], sequence: list[FrameSignals]) -> dict:
    segments = {segment: 0 for segment in {frame.segment for frame in sequence}}
    parallax = []
    for index in keyframes[1:]:
        previous = max(value for value in keyframes if value < index)
        metrics = relative_metrics(sequence[index], sequence[previous])
        segments[sequence[index].segment] += 1
        parallax.append(metrics["parallax_deg"])
    gaps = np.diff(keyframes) if len(keyframes) > 1 else np.array([FRAMES])
    return {
        "policy": name,
        "keyframes": len(keyframes),
        "minimum_spacing_frames": int(np.min(gaps)),
        "maximum_spacing_frames": int(np.max(gaps)),
        "median_insertion_parallax_deg": float(np.median(parallax)) if parallax else 0.0,
        "pure_rotation_insertions": segments["pure_rotation"],
        "forward_low_parallax_insertions": segments["forward_low_parallax"],
        "weak_texture_stop_insertions": segments["weak_texture_stop"],
        "indices": keyframes,
    }


def render_plot(decisions: list[dict], keyframes: list[int]) -> np.ndarray:
    width, height = 1280, 720
    canvas = np.full((height, width, 3), 246, dtype=np.uint8)
    left, right, top, bottom = 75, 1235, 80, 620
    colors = {
        "lateral_a": (236, 248, 244),
        "pure_rotation": (244, 239, 226),
        "forward_low_parallax": (239, 239, 249),
        "lateral_b": (236, 248, 244),
        "weak_texture_stop": (235, 235, 235),
        "lateral_recovery": (236, 248, 244),
    }
    start = 0
    current = decisions[0]["segment"]
    for index, row in enumerate(decisions + [{"segment": "end"}]):
        if row["segment"] != current:
            x0 = left + int((right - left) * start / (FRAMES - 1))
            x1 = left + int((right - left) * index / (FRAMES - 1))
            cv2.rectangle(canvas, (x0, top), (x1, bottom), colors[current], -1)
            cv2.putText(canvas, current.replace("_", " "), (x0 + 5, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (75, 80, 95), 1, cv2.LINE_AA)
            if index < FRAMES:
                start, current = index, row["segment"]
    cv2.line(canvas, (left, bottom), (right, bottom), (75, 80, 95), 2)
    tracks = np.array([row["tracked_features"] / 260.0 for row in decisions])
    parallax = np.array([min(row["parallax_deg"] / 2.0, 1.0) for row in decisions])
    compute = np.array([row["compute_load"] for row in decisions])
    for values, color, label, offset in [
        (tracks, (185, 115, 20), "tracked / 260", 0),
        (parallax, (50, 155, 115), "parallax / 2 deg", 22),
        (compute, (55, 70, 205), "compute load", 44),
    ]:
        points = []
        for index, value in enumerate(values):
            x = left + int((right - left) * index / (FRAMES - 1))
            y = bottom - int(value * (bottom - top - 50))
            points.append((x, y))
        cv2.polylines(canvas, [np.array(points)], False, color, 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (left + 10, bottom + 35 + offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    for index in keyframes:
        x = left + int((right - left) * index / (FRAMES - 1))
        cv2.line(canvas, (x, top), (x, bottom), (35, 35, 35), 2)
        cv2.circle(canvas, (x, top + 20), 5, (35, 35, 35), -1)
    cv2.putText(canvas, "MULTI-SIGNAL KEYFRAME POLICY", (left, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (35, 45, 65), 2, cv2.LINE_AA)
    return canvas


def main() -> int:
    OUT.mkdir(exist_ok=True)
    sequence = make_sequence()
    decisions, full_keyframes = run_full_policy(sequence)
    policies = {"full_policy": full_keyframes}
    for mode in ("rotation_only", "translation_only", "timer_only"):
        policies[mode] = run_ablation(sequence, mode)
    summaries = {name: summarize(name, values, sequence) for name, values in policies.items()}
    full = summaries["full_policy"]
    compute_blocks = sum(row["reason"] == "blocked_compute_budget" for row in decisions)
    weak_texture_defers = sum(row["reason"] == "defer_weak_texture" for row in decisions)
    low_parallax_defers = sum(row["reason"] == "defer_low_parallax" for row in decisions)
    report = {
        "frames": FRAMES,
        "dt_s": DT,
        "duration_s": (FRAMES - 1) * DT,
        "minimum_spacing_frames": MIN_SPACING_FRAMES,
        "policy": {
            "geometry": "lateral >= 0.10 m and parallax >= 0.90 deg",
            "view_change": "rotation >= 25 deg and overlap <= 0.70",
            "track_pressure": "90 <= tracks < 125 with parallax or rotation support",
            "timeout": "elapsed >= 3 s with parallax or rotation support",
            "guards": "minimum spacing, compute load < 0.80, tracks >= 90, two-frame hysteresis",
        },
        "full_policy": full,
        "ablations": {name: value for name, value in summaries.items() if name != "full_policy"},
        "blocked_compute_budget_frames": compute_blocks,
        "weak_texture_defer_frames": weak_texture_defers,
        "low_parallax_defer_frames": low_parallax_defers,
        "command": "python3 scripts/keyframe_policy_smoke.py",
        "artifacts": {
            "decision_log": "out/keyframe_decisions.csv",
            "ablation_table": "out/keyframe_ablation.csv",
            "model": "out/keyframe_policy_model.npz",
            "plot": "out/keyframe_policy_plot.png",
        },
    }
    checks = {
        "bounded_keyframe_count": 8 <= full["keyframes"] <= 16,
        "minimum_spacing_enforced": full["minimum_spacing_frames"] >= MIN_SPACING_FRAMES,
        "rotation_not_over_keyframed": full["pure_rotation_insertions"] <= 3,
        "rotation_ablation_exposes_failure": summaries["rotation_only"]["pure_rotation_insertions"] >= full["pure_rotation_insertions"] + 2,
        "forward_degeneracy_deferred": full["forward_low_parallax_insertions"] <= 1,
        "translation_ablation_exposes_failure": summaries["translation_only"]["forward_low_parallax_insertions"] >= 4,
        "weak_texture_not_keyframed": full["weak_texture_stop_insertions"] == 0 and weak_texture_defers >= 10,
        "compute_budget_active": compute_blocks >= 5,
        "hysteresis_recorded": sum(row["reason"] == "hysteresis_arming" for row in decisions) >= 5,
    }
    report["checks"] = checks
    report["status"] = "PASS" if all(checks.values()) else "FAIL"

    with (OUT / "keyframe_decisions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(decisions[0]))
        writer.writeheader()
        writer.writerows(decisions)
    with (OUT / "keyframe_ablation.csv").open("w", newline="") as handle:
        fields = [key for key in next(iter(summaries.values())) if key != "indices"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for value in summaries.values():
            writer.writerow({key: value[key] for key in fields})
    np.savez(
        OUT / "keyframe_policy_model.npz",
        positions_m=np.array([frame.position_m for frame in sequence]),
        yaw_deg=np.array([frame.yaw_deg for frame in sequence]),
        full_keyframes=np.array(full_keyframes),
        rotation_only_keyframes=np.array(policies["rotation_only"]),
        translation_only_keyframes=np.array(policies["translation_only"]),
        timer_only_keyframes=np.array(policies["timer_only"]),
    )
    cv2.imwrite(str(OUT / "keyframe_policy_plot.png"), render_plot(decisions, full_keyframes))
    (OUT / "keyframe_policy_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Keyframe policy lab:", report["status"])
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
