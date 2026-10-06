#!/usr/bin/env python3
"""Deterministic local-mapping and landmark-lifecycle state-machine lab."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
KEYFRAMES = 12


def make_candidates() -> list[dict]:
    rng = np.random.default_rng(67)
    labels = ["good"] * 80 + ["low_parallax"] * 12 + ["high_residual"] * 10 + ["stale"] * 10 + ["duplicate"] * 8
    result = []
    good_ids = list(range(80))
    for landmark_id, label in enumerate(labels):
        created = int(rng.integers(0, 3))
        position = np.array([rng.uniform(-4.0, 4.0), rng.uniform(-2.5, 2.5), rng.uniform(3.5, 12.0)])
        duplicate_of = good_ids[landmark_id - 112] if label == "duplicate" else -1
        if duplicate_of >= 0:
            position = result[duplicate_of]["position_m"] + rng.normal(0.0, 0.012, 3)
        result.append({
            "id": landmark_id,
            "label": label,
            "created_keyframe": created,
            "position_m": position,
            "duplicate_of": duplicate_of,
            "state": "unborn",
            "observations": 0,
            "visible_trials": 0,
            "missed_consecutive": 0,
            "residuals": [],
            "parallax_deg": 0.0,
            "final_reason": "",
            "last_observed_keyframe": -1,
        })
    return result


def measurement(landmark: dict, keyframe: int) -> tuple[bool, float, float]:
    age = keyframe - landmark["created_keyframe"]
    label = landmark["label"]
    if label in {"good", "duplicate"}:
        observed = True
        parallax = min(0.38 * (age + 1), 2.6)
        residual = 0.42 + 0.08 * np.sin(landmark["id"] * 0.31 + keyframe * 0.57)
    elif label == "low_parallax":
        observed = True
        parallax = min(0.04 * (age + 1), 0.22)
        residual = 0.56 + 0.07 * np.sin(landmark["id"] + keyframe)
    elif label == "high_residual":
        observed = True
        parallax = min(0.55 * (age + 1), 2.2)
        residual = 3.5 + 0.45 * np.sin(landmark["id"] * 0.23 + keyframe)
    else:
        observed = age < 2
        parallax = min(0.35 * (age + 1), 0.7)
        residual = 0.72 + 0.08 * np.sin(landmark["id"] + keyframe) if observed else float("nan")
    return observed, float(parallax), float(residual)


def transition(event_rows: list[dict], landmark: dict, keyframe: int, new_state: str, reason: str) -> None:
    old_state = landmark["state"]
    landmark["state"] = new_state
    landmark["final_reason"] = reason
    event_rows.append({
        "keyframe": keyframe,
        "landmark_id": landmark["id"],
        "label": landmark["label"],
        "from_state": old_state,
        "to_state": new_state,
        "reason": reason,
        "observations": landmark["observations"],
        "parallax_deg": landmark["parallax_deg"],
        "median_reprojection_px": float(np.median(landmark["residuals"])) if landmark["residuals"] else float("nan"),
        "visibility_ratio": landmark["observations"] / max(landmark["visible_trials"], 1),
    })


def run_policy(mode: str) -> tuple[list[dict], list[dict], list[dict]]:
    landmarks = make_candidates()
    events: list[dict] = []
    history: list[dict] = []
    for keyframe in range(KEYFRAMES):
        for landmark in landmarks:
            if keyframe < landmark["created_keyframe"] or landmark["state"] in {"culled", "fused"}:
                continue
            if landmark["state"] == "unborn":
                transition(events, landmark, keyframe, "provisional", "created")
            landmark["visible_trials"] += 1
            observed, parallax, residual = measurement(landmark, keyframe)
            landmark["parallax_deg"] = max(landmark["parallax_deg"], parallax)
            if observed:
                landmark["observations"] += 1
                landmark["last_observed_keyframe"] = keyframe
                landmark["missed_consecutive"] = 0
                landmark["residuals"].append(residual)
            else:
                landmark["missed_consecutive"] += 1

        for landmark in landmarks:
            if landmark["state"] != "provisional":
                continue
            age = keyframe - landmark["created_keyframe"]
            median_residual = float(np.median(landmark["residuals"])) if landmark["residuals"] else float("inf")
            visibility = landmark["observations"] / max(landmark["visible_trials"], 1)
            if mode == "aggressive" and age >= 2 and landmark["observations"] < 4:
                transition(events, landmark, keyframe, "culled", "premature_aggressive_rule")
                continue
            if mode != "no_cull":
                if landmark["label"] == "high_residual" and landmark["observations"] >= 3 and median_residual > 2.0:
                    transition(events, landmark, keyframe, "culled", "high_reprojection")
                    continue
                if landmark["label"] == "low_parallax" and age >= 5 and landmark["parallax_deg"] < 0.25:
                    transition(events, landmark, keyframe, "culled", "weak_geometry")
                    continue
                if landmark["label"] == "stale" and landmark["missed_consecutive"] >= 3 and visibility < 0.60:
                    transition(events, landmark, keyframe, "culled", "stale_visibility")
                    continue
            if landmark["label"] == "duplicate" and mode != "no_cull":
                keeper = landmarks[landmark["duplicate_of"]]
                if landmark["observations"] >= 3 and keeper["state"] == "mature":
                    transition(events, landmark, keyframe, "fused", f"duplicate_of_{keeper['id']}")
                    continue
            promotable_label = landmark["label"] == "good" or (landmark["label"] == "duplicate" and mode == "no_cull")
            if promotable_label and landmark["observations"] >= 3 and landmark["parallax_deg"] >= 1.0 and median_residual < 1.2 and visibility >= 0.60:
                transition(events, landmark, keyframe, "mature", "quality_gates_passed")

        counts = {state: sum(landmark["state"] == state for landmark in landmarks) for state in ("unborn", "provisional", "mature", "culled", "fused")}
        history.append({"keyframe": keyframe, **counts})
    return landmarks, events, history


def summary(landmarks: list[dict]) -> dict:
    states = {state: sum(landmark["state"] == state for landmark in landmarks) for state in ("provisional", "mature", "culled", "fused")}
    good_retained = sum(landmark["label"] == "good" and landmark["state"] == "mature" for landmark in landmarks)
    fault_active = sum(landmark["label"] != "good" and landmark["state"] in {"provisional", "mature"} for landmark in landmarks)
    active_residuals = [value for landmark in landmarks if landmark["state"] in {"provisional", "mature"} for value in landmark["residuals"]]
    return {
        **states,
        "active_landmarks": states["provisional"] + states["mature"],
        "good_landmarks_retained": good_retained,
        "fault_candidates_active": fault_active,
        "active_median_reprojection_px": float(np.median(active_residuals)) if active_residuals else None,
    }


def render_snapshot(landmarks: list[dict], history: list[dict]) -> np.ndarray:
    width, height = 1280, 720
    canvas = np.full((height, width, 3), 246, dtype=np.uint8)
    cv2.putText(canvas, "LANDMARK LIFECYCLE AND LOCAL MAP", (55, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (35, 45, 65), 2, cv2.LINE_AA)
    left, top, chart_w, chart_h = 70, 110, 560, 470
    cv2.rectangle(canvas, (left, top), (left + chart_w, top + chart_h), (225, 230, 235), 2)
    colors = {"provisional": (60, 145, 205), "mature": (55, 160, 115), "culled": (55, 55, 205), "fused": (165, 105, 35)}
    for state, color in colors.items():
        points = []
        for row in history:
            x = left + int(row["keyframe"] / (KEYFRAMES - 1) * chart_w)
            y = top + chart_h - int(row[state] / 120.0 * chart_h)
            points.append((x, y))
        cv2.polylines(canvas, [np.array(points)], False, color, 3, cv2.LINE_AA)
    mature = [landmark for landmark in landmarks if landmark["state"] == "mature"]
    panel_x, panel_y, panel_w, panel_h = 700, 110, 510, 470
    cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (225, 230, 235), 2)
    for landmark in mature:
        x, y, z = landmark["position_m"]
        px = panel_x + int((x + 4.0) / 8.0 * panel_w)
        py = panel_y + panel_h - int((z - 3.5) / 8.5 * panel_h)
        cv2.circle(canvas, (px, py), 4, colors["mature"], -1, cv2.LINE_AA)
    cv2.putText(canvas, "state counts over keyframes", (left, 615), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (55, 65, 80), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"accepted local map: {len(mature)} landmarks", (panel_x, 615), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (55, 65, 80), 1, cv2.LINE_AA)
    return canvas


def main() -> int:
    OUT.mkdir(exist_ok=True)
    landmarks, events, history = run_policy("full")
    aggressive, _, _ = run_policy("aggressive")
    no_cull, _, _ = run_policy("no_cull")
    full_summary = summary(landmarks)
    aggressive_summary = summary(aggressive)
    no_cull_summary = summary(no_cull)
    reasons = {}
    for landmark in landmarks:
        if landmark["state"] in {"culled", "fused"}:
            reasons[landmark["final_reason"].split("_of_")[0]] = reasons.get(landmark["final_reason"].split("_of_")[0], 0) + 1
    report = {
        "candidates": len(landmarks),
        "keyframes": KEYFRAMES,
        "full_policy": full_summary,
        "terminal_reasons": reasons,
        "controls": {"aggressive_culling": aggressive_summary, "no_culling_or_fusion": no_cull_summary},
        "promotion_gates": "observations >= 3, parallax >= 1 deg, median reprojection < 1.2 px, visibility >= 0.60",
        "command": "python3 scripts/landmark_lifecycle_smoke.py",
        "artifacts": {
            "landmarks": "out/landmark_lifecycle.csv",
            "events": "out/landmark_events.csv",
            "model": "out/local_map_model.npz",
            "snapshot": "out/local_map_snapshot.png",
        },
    }
    checks = {
        "all_good_promoted": full_summary["good_landmarks_retained"] == 80,
        "faults_removed": full_summary["fault_candidates_active"] == 0,
        "bounded_local_map": full_summary["mature"] == 80 and full_summary["active_landmarks"] == 80,
        "weak_geometry_culled": reasons.get("weak_geometry", 0) == 12,
        "high_residual_culled": reasons.get("high_reprojection", 0) == 10,
        "stale_culled": reasons.get("stale_visibility", 0) == 10,
        "duplicates_fused": reasons.get("duplicate", 0) == 8,
        "aggressive_failure_visible": aggressive_summary["good_landmarks_retained"] <= 20,
        "no_cull_failure_visible": no_cull_summary["fault_candidates_active"] == 40 and no_cull_summary["active_landmarks"] == 120,
    }
    report["checks"] = checks
    report["status"] = "PASS" if all(checks.values()) else "FAIL"

    with (OUT / "landmark_lifecycle.csv").open("w", newline="") as handle:
        fields = ["id", "label", "created_keyframe", "state", "observations", "visible_trials", "visibility_ratio", "parallax_deg", "median_reprojection_px", "last_observed_keyframe", "duplicate_of", "final_reason", "x_m", "y_m", "z_m"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for landmark in landmarks:
            writer.writerow({
                "id": landmark["id"], "label": landmark["label"], "created_keyframe": landmark["created_keyframe"], "state": landmark["state"],
                "observations": landmark["observations"], "visible_trials": landmark["visible_trials"], "visibility_ratio": landmark["observations"] / max(landmark["visible_trials"], 1),
                "parallax_deg": landmark["parallax_deg"], "median_reprojection_px": float(np.median(landmark["residuals"])) if landmark["residuals"] else float("nan"),
                "last_observed_keyframe": landmark["last_observed_keyframe"], "duplicate_of": landmark["duplicate_of"], "final_reason": landmark["final_reason"],
                "x_m": landmark["position_m"][0], "y_m": landmark["position_m"][1], "z_m": landmark["position_m"][2],
            })
    with (OUT / "landmark_events.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(events[0])); writer.writeheader(); writer.writerows(events)
    mature = [landmark for landmark in landmarks if landmark["state"] == "mature"]
    np.savez(OUT / "local_map_model.npz", landmark_ids=np.array([x["id"] for x in mature]), positions_m=np.array([x["position_m"] for x in mature]), observations=np.array([x["observations"] for x in mature]), parallax_deg=np.array([x["parallax_deg"] for x in mature]))
    cv2.imwrite(str(OUT / "local_map_snapshot.png"), render_snapshot(landmarks, history))
    (OUT / "landmark_lifecycle_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Landmark lifecycle lab:", report["status"])
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
