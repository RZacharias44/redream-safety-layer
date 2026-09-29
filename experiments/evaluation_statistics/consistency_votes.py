"""Derive the semantic-consistency verdict from the stored per-segment votes.

Why this module exists
----------------------
``experiments/run_benchmark.py`` always runs the full critic pipeline first and
writes its temperature-0 output to ``pred_safe``/``pred_node``. In consistency
mode it then samples each classified segment ``N`` times at temperature 0.7 and
stores those votes in ``vote_distribution``, but it never feeds the votes back
into triage. Scoring ``pred_safe`` for a consistency run therefore measures a
second deterministic run, not the majority-vote decision rule.

This module reconstructs the decision rule offline: it maps each segment's
majority vote to an ontology node and applies the same three-bucket triage the
critic uses, so the consistency conditions report the rule they describe. No
model calls are needed — every input is already in the checked-in CSVs.

Only the standard library is used, matching ``recompute_chapter4.py`` so the
reported counts stay auditable without a full runtime environment. The three
constants below are copies of the runtime definitions; ``tests/
test_consistency_votes.py`` asserts they stay in sync with ``safety/``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

# Mirror of safety.classifier.CODE_MAP (checked by tests).
CODE_MAP = {
    "HIDE": "AVOIDANCE",
    "ESCAPE": "INTERRUPTION",
    "DESTROY": "VIOLENT_REVENGE",
    "DENY": "SUPPRESSION",
    "REPLAY": "TRAUMA_REPLAY",
    "CONFRONT": "BEHAVIORAL_MASTERY",
    "HELP": "SOCIAL_MASTERY",
    "CHANGE": "ENVIRONMENTAL_MASTERY",
    "RELAX": "EMOTIONAL_MASTERY",
    "POWER": "MYTHICAL_MASTERY",
    "LOOK": "NARRATIVE_SETTING",
    "FEEL": "AFFECT_EXPRESSION",
}

# Mirror of safety.critic.SEVERITY_PRIORITY (checked by tests).
SEVERITY_PRIORITY = {
    "VIOLENT_REVENGE": 1,
    "TRAUMA_REPLAY": 1,
    "INTERRUPTION": 2,
    "SUPPRESSION": 3,
    "AVOIDANCE": 4,
}

MALADAPTIVE_NODES = frozenset(SEVERITY_PRIORITY)
NEUTRAL_NODES = frozenset({"NARRATIVE_SETTING", "AFFECT_EXPRESSION"})

_SEGMENT_TYPE = re.compile(r"\[([A-Z_]+)\]")
_SEGMENT_NODE = re.compile(r"→ (\S+) \(")
_SEGMENT_SPLIT = re.compile(r"\s\|\|?\s")


def parse_segments(cell: str | None) -> list[tuple[str, str | None]]:
    """``segments`` column → ordered ``(segment_type, node)`` pairs.

    ``node`` is ``None`` for rows written by ``topup_s1_consistency.py``, whose
    formatter omits the ``→ NODE (conf)`` suffix.
    """
    segments: list[tuple[str, str | None]] = []
    for part in _SEGMENT_SPLIT.split(cell or ""):
        type_match = _SEGMENT_TYPE.search(part)
        if not type_match:
            continue
        node_match = _SEGMENT_NODE.search(part)
        segments.append((type_match.group(1), node_match.group(1) if node_match else None))
    return segments


def majority_node(votes: dict[str, int]) -> tuple[str, float]:
    """One segment's vote counts → ``(majority node, agreement ratio)``.

    Ties resolve to the first key, reproducing ``Counter.most_common`` on the
    order ``run_benchmark.py`` serialized.
    """
    total = sum(votes.values())
    code, count = max(votes.items(), key=lambda item: item[1])
    return CODE_MAP.get(code, code), count / total


def triage(nodes: list[str], has_unclassified: bool) -> tuple[bool, str]:
    """Three-bucket triage from ``SafetyCritic.evaluate_intervention``.

    Maladaptive outranks unclassified, which outranks adaptive/neutral. Among
    maladaptive segments the highest severity wins, ties going to the earliest
    segment.
    """
    maladaptive = [node for node in nodes if node in MALADAPTIVE_NODES]
    if maladaptive:
        primary = min(
            range(len(maladaptive)),
            key=lambda i: (SEVERITY_PRIORITY[maladaptive[i]], i),
        )
        return False, maladaptive[primary]
    if has_unclassified:
        return False, "UNCLASSIFIED"
    adaptive = [node for node in nodes if node not in NEUTRAL_NODES]
    if adaptive:
        return True, adaptive[0]
    return True, nodes[0] if nodes else "UNKNOWN"


def _unclassified_present(segments: list[tuple[str, str | None]]) -> bool:
    """True when a non-META segment reached triage without a usable node.

    Such segments get no votes, so the vote-based verdict inherits the critic's
    fail-closed handling. None occur in the development set.
    """
    return any(
        seg_type != "META" and node in {"?", "UNCLASSIFIED"}
        for seg_type, node in segments
    )


def derive_vote_verdicts(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Add ``vote_safe``/``vote_node`` to consistency rows.

    Returns the annotated rows plus diagnostics: how many rows fell back to the
    pipeline verdict (``META_ONLY`` cases have no classified segment to vote
    on), how many verdicts the votes flipped, and whether the reconstruction
    reproduces the stored ``agreement_ratio``/``consistency_node``.
    """
    annotated: list[dict[str, Any]] = []
    fallback = flips = roundtrip_ok = roundtrip_total = 0

    for row in rows:
        row = dict(row)
        raw_votes = row.get("vote_distribution")
        if not raw_votes:
            row["vote_safe"] = row.get("pred_safe")
            row["vote_node"] = row.get("pred_node")
            fallback += 1
            annotated.append(row)
            continue

        groups = json.loads(raw_votes)
        per_segment = {key: majority_node(votes) for key, votes in groups.items()}
        segments = parse_segments(row.get("segments"))
        safe, node = triage(
            [node for node, _ in per_segment.values()],
            _unclassified_present(segments),
        )
        row["vote_safe"] = "True" if safe else "False"
        row["vote_node"] = node
        flips += str(row.get("pred_safe")) != row["vote_safe"]

        # Round-trip check: the weakest segment must match the stored fields.
        weakest = min(per_segment, key=lambda key: per_segment[key][1])
        stored_ratio = row.get("agreement_ratio")
        if stored_ratio:
            roundtrip_total += 1
            roundtrip_ok += (
                abs(per_segment[weakest][1] - float(stored_ratio)) < 1e-9
                and per_segment[weakest][0] == row.get("consistency_node")
            )
        annotated.append(row)

    # Agreement spread over the scored cases: reported in the thesis as evidence
    # that temperature 0.7 leaves the vote neither saturated nor degenerate.
    scored_agreements = [
        float(row["agreement_ratio"])
        for row in annotated
        if row.get("agreement_ratio") and row.get("gt_safe") in ("True", "False")
    ]
    unanimous = sum(value >= 1.0 for value in scored_agreements)

    return annotated, {
        "rows": len(annotated),
        "pipeline_fallback_rows": fallback,
        "verdict_flips_vs_pipeline": flips,
        "roundtrip_matches": roundtrip_ok,
        "roundtrip_checked": roundtrip_total,
        "scored_rows_with_agreement": len(scored_agreements),
        "unanimous_rows": unanimous,
        "unanimous_share": (
            round(unanimous / len(scored_agreements), 6) if scored_agreements else None
        ),
        "mean_agreement_ratio": (
            round(sum(scored_agreements) / len(scored_agreements), 6)
            if scored_agreements
            else None
        ),
        "min_agreement_ratio": min(scored_agreements) if scored_agreements else None,
    }


def verify_pipeline_triage(rows: Iterable[dict[str, Any]]) -> tuple[int, int, int]:
    """Validate the triage re-implementation against the critic's own output.

    Feeds each row's *pipeline* segment nodes through :func:`triage` and counts
    how often that reproduces ``pred_safe`` and ``pred_node``. Rows whose
    ``segments`` cell lacks per-segment nodes are skipped.

    Returns ``(reproduced, checked, skipped)``.
    """
    reproduced = checked = skipped = 0
    for row in rows:
        segments = parse_segments(row.get("segments"))
        if not segments or any(node is None for _type, node in segments):
            skipped += 1
            continue
        nodes = [
            node for seg_type, node in segments
            if seg_type != "META" and node not in {"?", "UNCLASSIFIED"}
        ]
        unclassified = _unclassified_present(segments)
        if not nodes and not unclassified:
            skipped += 1
            continue
        safe, node = triage(nodes, unclassified)
        checked += 1
        reproduced += (
            ("True" if safe else "False") == str(row.get("pred_safe"))
            and node == row.get("pred_node")
        )
    return reproduced, checked, skipped
