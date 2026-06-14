from __future__ import annotations

import json
import math
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table

from tarmac.defect import (
    DEFECT_LABELS,
    DEFECT_LABEL_APPLICABILITY,
    STRUCTURAL_DEFECT_DOMAINS,
    infer_defect_domain,
    is_defect_label_applicable,
)
from tarmac.inference.analyze import analyze_path

SEED = 42

PCI_PROXY_DISCLAIMER = (
    "PCI-like descriptor is a transparent visual proxy, not an official ASTM D6433 PCI."
)
VISUAL_LIMITATION_NOTICE = (
    "Binder content, density/air voids, and water-damage or stripping progression are not "
    "measured; this assessment uses visible image proxies only."
)

PCI_PROXY_DESCRIPTORS = {
    1: "Good",
    2: "Satisfactory",
    3: "Fair",
    4: "Poor",
    5: "Serious",
}

# Higher grade means worse condition, aligned to the project quality scale.
# These are proxy aggregation rules over existing analyze outputs, not ASTM D6433 PCI formulas.
CONDITION_GRADE_RULES: list[dict[str, Any]] = [
    {
        "id": "quality_baseline",
        "grade_floor": "quality_grade",
        "description": "Start from the visual surface quality grade when available.",
    },
    {
        "id": "critical_structural_defect",
        "grade_floor": 5,
        "any_defects": ["exposed_rebar", "corrosion"],
        "domains": ["bridge", "building"],
        "description": "Exposed reinforcement or corrosion on inferred bridge/building imagery.",
    },
    {
        "id": "wide_crack",
        "grade_floor": 5,
        "crack_width_band": "wide",
        "description": "AASHTO-inspired crack width band is wide (>3.2 mm).",
    },
    {
        "id": "large_crack_area",
        "grade_floor": 5,
        "crack_area_pct_min": 5.0,
        "description": "Crack mask covers at least 5% of the image.",
    },
    {
        "id": "spalling",
        "grade_floor": 4,
        "any_defects": ["spalling"],
        "description": "Spalling is present in at least one analyzed tile.",
    },
    {
        "id": "moderate_crack",
        "grade_floor": 4,
        "crack_width_band": "moderate",
        "description": "AASHTO-inspired crack width band is moderate (1.6-3.2 mm).",
    },
    {
        "id": "material_crack_area",
        "grade_floor": 4,
        "crack_area_pct_min": 1.0,
        "description": "Crack mask covers at least 1% of the image.",
    },
    {
        "id": "many_cracked_tiles",
        "grade_floor": 4,
        "crack_ratio_min": 0.5,
        "description": "At least half of road tiles are crack-positive.",
    },
    {
        "id": "minor_crack",
        "grade_floor": 3,
        "any_defects": ["crack"],
        "description": "Crack signal is present.",
    },
    {
        "id": "efflorescence",
        "grade_floor": 3,
        "any_defects": ["efflorescence"],
        "description": "Efflorescence is present.",
    },
]

# Ordered from highest to lowest urgency. First matching priority level wins, while all
# matching rule IDs are kept in the assessment record for transparency.
REPAIR_PRIORITY_RULES: list[dict[str, Any]] = [
    {
        "priority": "urgent",
        "id": "structural_exposed_rebar_or_corrosion",
        "any_defects": ["exposed_rebar", "corrosion"],
        "domains": ["bridge", "building"],
        "description": "Exposed rebar or corrosion on inferred bridge/building imagery.",
    },
    {
        "priority": "urgent",
        "id": "wide_crack",
        "crack_width_band": "wide",
        "description": "AASHTO-inspired wide crack band (>3.2 mm).",
    },
    {
        "priority": "urgent",
        "id": "quality_grade_5",
        "quality_min": 5,
        "description": "Visual quality grade is 5.",
    },
    {
        "priority": "urgent",
        "id": "large_crack_area",
        "crack_area_pct_min": 5.0,
        "description": "Crack mask covers at least 5% of the image.",
    },
    {
        "priority": "plan_repair",
        "id": "moderate_crack",
        "crack_width_band": "moderate",
        "description": "AASHTO-inspired moderate crack band (1.6-3.2 mm).",
    },
    {
        "priority": "plan_repair",
        "id": "spalling",
        "any_defects": ["spalling"],
        "description": "Spalling is present.",
    },
    {
        "priority": "plan_repair",
        "id": "quality_grade_4",
        "quality_min": 4,
        "description": "Visual quality grade is 4.",
    },
    {
        "priority": "plan_repair",
        "id": "material_crack_area",
        "crack_area_pct_min": 1.0,
        "description": "Crack mask covers at least 1% of the image.",
    },
    {
        "priority": "plan_repair",
        "id": "many_cracked_tiles",
        "crack_ratio_min": 0.5,
        "description": "At least half of road tiles are crack-positive.",
    },
    {
        "priority": "monitor",
        "id": "minor_crack",
        "any_defects": ["crack"],
        "description": "Crack signal is present below repair thresholds.",
    },
    {
        "priority": "monitor",
        "id": "efflorescence",
        "any_defects": ["efflorescence"],
        "description": "Efflorescence is present without higher-priority structural triggers.",
    },
    {
        "priority": "monitor",
        "id": "quality_grade_3",
        "quality_min": 3,
        "quality_max": 3,
        "description": "Visual quality grade is 3, treated as early wear/raveling proxy.",
    },
]

REPAIR_PRIORITY_ORDER = {"none": 0, "monitor": 1, "plan_repair": 2, "urgent": 3}
STRUCTURAL_DOMAINS = STRUCTURAL_DEFECT_DOMAINS


def assess_path(
    input_path: Path,
    out_dir: Path | None = None,
    fps: float = 2.0,
    k: int = 10,
    non_road_threshold: float | None = None,
    batch_size: int = 16,
    device: str = "cpu",
    region: str = "auto",
    mm_per_pixel: float | None = None,
    defect_gating: bool = True,
) -> dict[str, Any]:
    """Run analyze, then aggregate visual condition and repair-priority proxy records."""
    _seed_everything(SEED)
    input_path = input_path.expanduser().resolve()
    if out_dir is None:
        out_dir = Path("runs") / f"{input_path.stem}_assessment"
    out_dir = out_dir.expanduser().resolve()

    analysis_summary = analyze_path(
        input_path=input_path,
        out_dir=out_dir,
        fps=fps,
        k=k,
        non_road_threshold=non_road_threshold,
        batch_size=batch_size,
        device=device,
        region=region,
        crack_segmentation=True,
        mm_per_pixel=mm_per_pixel,
        defect_gating=defect_gating,
    )

    results_path = out_dir / "results.parquet"
    frames_df = pd.read_parquet(results_path)
    records = [
        condition_record(row=row, mm_per_pixel=mm_per_pixel)
        for row in frames_df.to_dict(orient="records")
    ]
    flat_rows = [_flatten_record(record) for record in records]
    assessment_df = pd.DataFrame(flat_rows)
    parquet_path = out_dir / "assessment.parquet"
    assessment_df.to_parquet(parquet_path, index=False)

    summary = condition_summary(records)
    json_path = out_dir / "assessment.json"
    payload = {
        "metadata": {
            "input_path": str(input_path),
            "out_dir": str(out_dir),
            "analysis_results_parquet": str(results_path),
            "assessment_parquet": str(parquet_path),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "seed": SEED,
            "device": device,
            "mm_per_pixel": mm_per_pixel,
            "defect_gating_enabled": bool(defect_gating),
            "defect_label_applicability": DEFECT_LABEL_APPLICABILITY,
            "pci_proxy_disclaimer": PCI_PROXY_DISCLAIMER,
            "visual_limitations": VISUAL_LIMITATION_NOTICE,
            "condition_grade_rules": CONDITION_GRADE_RULES,
            "repair_priority_rules": REPAIR_PRIORITY_RULES,
        },
        "analysis_summary": analysis_summary,
        "summary": {
            **summary,
            "assessment_json": str(json_path),
            "assessment_parquet": str(parquet_path),
        },
        "records": records,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def condition_record(row: dict[str, Any], mm_per_pixel: float | None = None) -> dict[str, Any]:
    surface_type = str(row.get("surface_type") or "unknown")
    quality_grade = _maybe_int(row.get("predicted_quality"))
    tiles = _tile_details(row.get("tile_details"))
    crack_geometry = _crack_geometry(row=row, mm_per_pixel=mm_per_pixel)
    inferred_domain = infer_domain(row=row, surface_type=surface_type)
    defects = _defect_signals(
        row=row,
        tiles=tiles,
        surface_type=surface_type,
        inferred_domain=inferred_domain,
    )
    context = {
        "quality_grade": quality_grade,
        "defects": defects,
        "crack_geometry": crack_geometry,
        "inferred_domain": inferred_domain,
        "crack_ratio": defects["crack"]["tile_ratio"],
    }
    overall_grade, condition_rules = overall_condition_grade(context)
    priority, repair_rules = compute_repair_priority(context)
    key_defects = [label for label, signal in defects.items() if signal["present"]]
    descriptor = PCI_PROXY_DESCRIPTORS[overall_grade]
    rationale = build_rationale(
        surface_type=surface_type,
        quality_grade=quality_grade,
        inferred_domain=inferred_domain,
        key_defects=key_defects,
        defects=defects,
        crack_geometry=crack_geometry,
        overall_grade=overall_grade,
        descriptor=descriptor,
        repair_priority=priority,
        condition_rules=condition_rules,
        repair_rules=repair_rules,
    )
    return {
        "frame_index": _maybe_int(row.get("frame_index")),
        "input_type": str(row.get("input_type") or ""),
        "source_path": str(row.get("source_path") or ""),
        "filename": str(row.get("filename") or ""),
        "thumbnail_path": str(row.get("thumbnail_path") or ""),
        "timestamp": _clean_scalar(row.get("timestamp")),
        "latitude": _maybe_float(row.get("latitude")),
        "longitude": _maybe_float(row.get("longitude")),
        "surface_type": surface_type,
        "inferred_domain": inferred_domain,
        "quality_grade": quality_grade,
        "confidence": _maybe_float(row.get("confidence")),
        "defect_gating_enabled": _defect_gating_enabled(row),
        "defects": defects,
        "crack_geometry": crack_geometry,
        "overall_condition_grade": overall_grade,
        "pci_proxy_descriptor": descriptor,
        "repair_priority": priority,
        "key_defects": key_defects,
        "applied_condition_rules": condition_rules,
        "applied_repair_rules": repair_rules,
        "rationale": rationale,
        "proxy_disclaimer": PCI_PROXY_DISCLAIMER,
        "visual_limitations": VISUAL_LIMITATION_NOTICE,
    }


def overall_condition_grade(context: dict[str, Any]) -> tuple[int, list[str]]:
    quality_grade = context.get("quality_grade")
    grade = int(quality_grade) if quality_grade is not None else 3
    matched: list[str] = []
    for rule in CONDITION_GRADE_RULES:
        if rule["id"] == "quality_baseline":
            if quality_grade is not None:
                matched.append(rule["id"])
            continue
        if _rule_matches(rule, context):
            grade = max(grade, int(rule["grade_floor"]))
            matched.append(str(rule["id"]))
    return min(max(grade, 1), 5), matched


def compute_repair_priority(context: dict[str, Any]) -> tuple[str, list[str]]:
    best = "none"
    matched: list[str] = []
    for rule in REPAIR_PRIORITY_RULES:
        if not _rule_matches(rule, context):
            continue
        matched.append(str(rule["id"]))
        priority = str(rule["priority"])
        if REPAIR_PRIORITY_ORDER[priority] > REPAIR_PRIORITY_ORDER[best]:
            best = priority
    return best, matched


def infer_domain(row: dict[str, Any], surface_type: str) -> str:
    return infer_defect_domain(
        source_path=str(row.get("source_path") or ""),
        filename=str(row.get("filename") or ""),
        surface_type=surface_type,
    )


def build_rationale(
    surface_type: str,
    quality_grade: int | None,
    inferred_domain: str,
    key_defects: list[str],
    defects: dict[str, dict[str, Any]],
    crack_geometry: dict[str, Any],
    overall_grade: int,
    descriptor: str,
    repair_priority: str,
    condition_rules: list[str],
    repair_rules: list[str],
) -> str:
    quality_text = f"quality grade {quality_grade}" if quality_grade is not None else "no quality vote"
    defect_text = ", ".join(
        f"{label} p={float(defects[label]['probability']):.2f}"
        for label in key_defects
        if defects[label].get("probability") is not None
    )
    if not defect_text:
        defect_text = "no positive defect flags"
    geometry_parts = []
    area_pct = crack_geometry.get("area_pct")
    if area_pct is not None:
        geometry_parts.append(f"crack area {float(area_pct):.3f}%")
    mean_width_mm = crack_geometry.get("mean_width_mm")
    if mean_width_mm is not None:
        geometry_parts.append(
            f"mean crack width {float(mean_width_mm):.2f} mm ({crack_geometry.get('aashto_width_band')})"
        )
    elif crack_geometry.get("mean_width_px") is not None:
        geometry_parts.append(f"mean crack width {float(crack_geometry['mean_width_px']):.2f} px")
    geometry_text = "; ".join(geometry_parts) if geometry_parts else "no crack geometry columns"
    condition_text = ", ".join(condition_rules) if condition_rules else "default proxy rules"
    repair_text = ", ".join(repair_rules) if repair_rules else "no repair trigger"
    return (
        f"{surface_type} ({inferred_domain}) with {quality_text}; {defect_text}; {geometry_text}. "
        f"Proxy grade {overall_grade} ({descriptor}) and priority {repair_priority} from "
        f"condition rules [{condition_text}] and repair rules [{repair_text}]. "
        f"{PCI_PROXY_DISCLAIMER} {VISUAL_LIMITATION_NOTICE}"
    )


def condition_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    priorities = Counter(str(record["repair_priority"]) for record in records)
    descriptors = Counter(str(record["pci_proxy_descriptor"]) for record in records)
    grades = [int(record["overall_condition_grade"]) for record in records]
    return {
        "frames_assessed": len(records),
        "mean_condition_grade": float(np.mean(grades)) if grades else 0.0,
        "repair_priority_counts": {priority: int(priorities.get(priority, 0)) for priority in REPAIR_PRIORITY_ORDER},
        "descriptor_counts": {name: int(descriptors.get(name, 0)) for name in PCI_PROXY_DESCRIPTORS.values()},
    }


def print_assessment_summary(payload: dict[str, Any], console: Console | None = None) -> None:
    console = console or Console()
    records = payload.get("records", [])
    table = Table(title="Tarmac Condition Assessment")
    table.add_column("Frame", justify="right")
    table.add_column("File")
    table.add_column("Surface")
    table.add_column("Q", justify="right")
    table.add_column("Condition", justify="right")
    table.add_column("Descriptor")
    table.add_column("Priority")
    table.add_column("Key defects")
    for record in records:
        key_defects = ", ".join(record.get("key_defects", [])) or "none"
        table.add_row(
            str(record.get("frame_index")),
            str(record.get("filename")),
            str(record.get("surface_type")),
            str(record.get("quality_grade") or "n/a"),
            str(record.get("overall_condition_grade")),
            str(record.get("pci_proxy_descriptor")),
            str(record.get("repair_priority")),
            key_defects,
        )
    console.print(table)
    summary = payload.get("summary", {})
    counts = ", ".join(
        f"{priority}={count}"
        for priority, count in summary.get("repair_priority_counts", {}).items()
    )
    console.print(
        f"Mean proxy condition grade: {float(summary.get('mean_condition_grade', 0.0)):.2f}; "
        f"repair priorities: {counts}"
    )
    console.print(f"{PCI_PROXY_DISCLAIMER} {VISUAL_LIMITATION_NOTICE}")


def repair_priority_rules_table(console: Console | None = None) -> None:
    console = console or Console()
    table = Table(title="Repair Priority Rules")
    table.add_column("Priority")
    table.add_column("Rule ID")
    table.add_column("Description")
    for rule in REPAIR_PRIORITY_RULES:
        table.add_row(str(rule["priority"]), str(rule["id"]), str(rule["description"]))
    console.print(table)


def _defect_signals(
    row: dict[str, Any],
    tiles: list[dict[str, Any]],
    surface_type: str,
    inferred_domain: str,
) -> dict[str, dict[str, Any]]:
    gating_enabled = _defect_gating_enabled(row)
    signals: dict[str, dict[str, Any]] = {}
    for label in DEFECT_LABELS:
        ratio_values: list[float] = []
        probability_values: list[float] = []
        raw_probability_values: list[float] = []
        applicable = True
        present = False
        if label == "crack":
            present = _maybe_bool(row.get("frame_has_crack")) or _maybe_bool(row.get("frame_has_defect_crack"))
            ratio_values.extend(
                value
                for value in (
                    _maybe_float(row.get("crack_ratio")),
                    _maybe_float(row.get("defect_crack_ratio")),
                )
                if value is not None
            )
            for tile in tiles:
                for value in (
                    _maybe_float(tile.get("tile_crack_prob")),
                    _maybe_float(tile.get("tile_defect_crack_prob")),
                    _maybe_float(tile.get("tile_defect_crack_prob_raw")),
                ):
                    if value is not None:
                        probability_values.append(value)
                        raw_probability_values.append(value)
        else:
            applicable = _frame_label_applicable(
                row=row,
                label=label,
                surface_type=surface_type,
                inferred_domain=inferred_domain,
                gating_enabled=gating_enabled,
            )
            tile_flags: list[bool] = []
            tile_applicable_seen = False
            for tile in tiles:
                raw_prob = _maybe_float(tile.get(f"tile_defect_{label}_prob_raw"))
                prob = _maybe_float(tile.get(f"tile_defect_{label}_prob"))
                if raw_prob is None:
                    raw_prob = prob
                if raw_prob is not None:
                    raw_probability_values.append(raw_prob)
                tile_applicable = _tile_label_applicable(
                    tile=tile,
                    label=label,
                    fallback_surface=surface_type,
                    fallback_domain=inferred_domain,
                    gating_enabled=gating_enabled,
                )
                if not tile_applicable:
                    continue
                tile_applicable_seen = True
                tile_flags.append(_maybe_bool(tile.get(f"tile_defect_{label}")))
                if prob is not None:
                    probability_values.append(prob)
            applicable = bool(applicable or tile_applicable_seen)
            if tile_flags:
                ratio_values.append(float(np.mean(tile_flags)))
                present = any(tile_flags)
            elif applicable:
                present = _maybe_bool(row.get(f"frame_has_defect_{label}"))
                ratio = _maybe_float(row.get(f"defect_{label}_ratio"))
                if ratio is not None:
                    ratio_values.append(ratio)
            else:
                present = False
            if applicable and not probability_values:
                prob = _maybe_float(row.get(f"defect_{label}_ratio"))
                if prob is not None:
                    probability_values.append(prob)
        tile_ratio = max(ratio_values) if ratio_values else 0.0
        probability = max(probability_values) if probability_values else tile_ratio
        raw_probability = max(raw_probability_values) if raw_probability_values else probability
        if not applicable:
            tile_ratio = 0.0
            probability = 0.0
            present = False
        signals[label] = {
            "present": bool(present or tile_ratio > 0.0),
            "probability": float(probability),
            "probability_raw": float(raw_probability),
            "tile_ratio": float(tile_ratio),
            "applicable": bool(applicable),
        }
    return signals


def _frame_label_applicable(
    row: dict[str, Any],
    label: str,
    surface_type: str,
    inferred_domain: str,
    gating_enabled: bool,
) -> bool:
    if not gating_enabled:
        return True
    return is_defect_label_applicable(
        label=label,
        surface_type=surface_type,
        domain=inferred_domain,
        source_path=str(row.get("source_path") or ""),
        filename=str(row.get("filename") or ""),
    )


def _tile_label_applicable(
    tile: dict[str, Any],
    label: str,
    fallback_surface: str,
    fallback_domain: str,
    gating_enabled: bool,
) -> bool:
    if not gating_enabled:
        return True
    explicit = tile.get(f"tile_defect_{label}_applicable")
    if explicit is not None:
        return _maybe_bool(explicit)
    return is_defect_label_applicable(
        label=label,
        surface_type=str(tile.get("surface_type") or fallback_surface),
        domain=str(tile.get("defect_domain") or fallback_domain),
    )


def _defect_gating_enabled(row: dict[str, Any]) -> bool:
    value = _clean_scalar(row.get("defect_gating_enabled"))
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _crack_geometry(row: dict[str, Any], mm_per_pixel: float | None) -> dict[str, Any]:
    area_px = _maybe_float(row.get("crack_area_px"))
    area_pct = _maybe_float(row.get("crack_area_pct"))
    length_px = _maybe_float(row.get("crack_length_px"))
    mean_width_px = _maybe_float(row.get("crack_mean_width_px"))
    max_width_px = _maybe_float(row.get("crack_max_width_px"))
    components = _maybe_int(row.get("crack_components"))
    area_mm2 = _maybe_float(row.get("crack_area_mm2"))
    length_mm = _maybe_float(row.get("crack_length_mm"))
    mean_width_mm = _maybe_float(row.get("crack_mean_width_mm"))
    max_width_mm = _maybe_float(row.get("crack_max_width_mm"))
    if mm_per_pixel is not None:
        if area_mm2 is None and area_px is not None:
            area_mm2 = area_px * (mm_per_pixel**2)
        if length_mm is None and length_px is not None:
            length_mm = length_px * mm_per_pixel
        if mean_width_mm is None and mean_width_px is not None:
            mean_width_mm = mean_width_px * mm_per_pixel
        if max_width_mm is None and max_width_px is not None:
            max_width_mm = max_width_px * mm_per_pixel
    return {
        "area_px": area_px,
        "area_pct": area_pct,
        "length_px": length_px,
        "mean_width_px": mean_width_px,
        "max_width_px": max_width_px,
        "components": components,
        "area_mm2": area_mm2,
        "length_mm": length_mm,
        "mean_width_mm": mean_width_mm,
        "max_width_mm": max_width_mm,
        "aashto_width_band": aashto_width_band(mean_width_mm),
    }


def aashto_width_band(width_mm: float | None) -> str | None:
    if width_mm is None:
        return None
    if width_mm < 1.6:
        return "narrow"
    if width_mm <= 3.2:
        return "moderate"
    return "wide"


def _rule_matches(rule: dict[str, Any], context: dict[str, Any]) -> bool:
    quality_grade = context.get("quality_grade")
    if "quality_min" in rule:
        if quality_grade is None or int(quality_grade) < int(rule["quality_min"]):
            return False
    if "quality_max" in rule:
        if quality_grade is None or int(quality_grade) > int(rule["quality_max"]):
            return False
    if "any_defects" in rule:
        defects = context.get("defects", {})
        if not any(bool(defects.get(label, {}).get("present", False)) for label in rule["any_defects"]):
            return False
    if "domains" in rule:
        if str(context.get("inferred_domain")) not in set(rule["domains"]):
            return False
    if "crack_width_band" in rule:
        band = context.get("crack_geometry", {}).get("aashto_width_band")
        if band != rule["crack_width_band"]:
            return False
    if "crack_area_pct_min" in rule:
        area_pct = context.get("crack_geometry", {}).get("area_pct")
        if area_pct is None or float(area_pct) < float(rule["crack_area_pct_min"]):
            return False
    if "crack_ratio_min" in rule:
        if float(context.get("crack_ratio") or 0.0) < float(rule["crack_ratio_min"]):
            return False
    return True


def _tile_details(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
    elif isinstance(value, list):
        parsed = value
    else:
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {
        "frame_index": record["frame_index"],
        "input_type": record["input_type"],
        "source_path": record["source_path"],
        "filename": record["filename"],
        "thumbnail_path": record["thumbnail_path"],
        "timestamp": record["timestamp"],
        "latitude": record["latitude"],
        "longitude": record["longitude"],
        "surface_type": record["surface_type"],
        "inferred_domain": record["inferred_domain"],
        "quality_grade": record["quality_grade"],
        "confidence": record["confidence"],
        "defect_gating_enabled": record["defect_gating_enabled"],
        "overall_condition_grade": record["overall_condition_grade"],
        "pci_proxy_descriptor": record["pci_proxy_descriptor"],
        "repair_priority": record["repair_priority"],
        "key_defects": ", ".join(record["key_defects"]),
        "applied_condition_rules": ", ".join(record["applied_condition_rules"]),
        "applied_repair_rules": ", ".join(record["applied_repair_rules"]),
        "rationale": record["rationale"],
        "proxy_disclaimer": record["proxy_disclaimer"],
        "visual_limitations": record["visual_limitations"],
        "defect_signals": json.dumps(record["defects"], sort_keys=True),
        "crack_geometry": json.dumps(record["crack_geometry"], sort_keys=True),
    }
    for label, signal in record["defects"].items():
        flat[f"defect_{label}"] = bool(signal["present"])
        flat[f"defect_{label}_prob"] = float(signal["probability"])
        flat[f"defect_{label}_prob_raw"] = float(signal["probability_raw"])
        flat[f"defect_{label}_tile_ratio"] = float(signal["tile_ratio"])
        flat[f"defect_{label}_applicable"] = bool(signal["applicable"])
    for key, value in record["crack_geometry"].items():
        flat[f"crack_{key}"] = value
    return flat


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def _maybe_int(value: Any) -> int | None:
    value = _clean_scalar(value)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_float(value: Any) -> float | None:
    value = _clean_scalar(value)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _maybe_bool(value: Any) -> bool:
    value = _clean_scalar(value)
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _clean_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value
