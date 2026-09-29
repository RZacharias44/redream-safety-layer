#!/usr/bin/env python3
"""Tests for the semantic-consistency vote derivation.

The consistency benchmark stores per-segment votes separately from the
temperature-0 pipeline output, so the majority-vote verdict has to be
reconstructed offline (``experiments/evaluation_statistics/consistency_votes.py``).
These tests guard the two ways that reconstruction could silently go wrong:

1. the local copies of ``CODE_MAP``/``SEVERITY_PRIORITY`` drifting from the
   runtime definitions in ``safety/``;
2. the re-implemented triage disagreeing with ``SafetyCritic``.

The second is checked against the real benchmark CSVs: feeding the pipeline's
own segment labels through the re-implementation must reproduce the verdict the
critic recorded for that row.
"""

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.evaluation_statistics.consistency_votes import (  # noqa: E402
    CODE_MAP,
    SEVERITY_PRIORITY,
    derive_vote_verdicts,
    majority_node,
    parse_segments,
    triage,
    verify_pipeline_triage,
)

REPO_ROOT = Path(__file__).parent.parent
CONSISTENCY_CSVS = {
    "s1": REPO_ROOT / "data/benchmarks/dev_s1_consistency_n5.csv",
    "s2": REPO_ROOT / "data/benchmarks/dev_s2_consistency_n5.csv",
}


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_constants_match_runtime_definitions():
    """The stdlib-only copies must stay in sync with safety/."""
    from safety.classifier import CODE_MAP as runtime_code_map
    from safety.critic import SEVERITY_PRIORITY as runtime_severity

    assert CODE_MAP == runtime_code_map
    assert SEVERITY_PRIORITY == runtime_severity


def test_code_map_is_one_to_one():
    """Code votes may stand in for node votes only if the mapping is a bijection."""
    assert len(set(CODE_MAP.values())) == len(CODE_MAP)


def test_majority_node_and_ties():
    assert majority_node({"HIDE": 4, "REPLAY": 1}) == ("AVOIDANCE", 0.8)
    assert majority_node({"CHANGE": 5}) == ("ENVIRONMENTAL_MASTERY", 1.0)
    # A tie resolves to the first key, as Counter.most_common serialized it.
    assert majority_node({"POWER": 2, "CHANGE": 2})[0] == "MYTHICAL_MASTERY"


def test_triage_prefers_highest_severity_maladaptive():
    safe, node = triage(["BEHAVIORAL_MASTERY", "AVOIDANCE", "VIOLENT_REVENGE"], False)
    assert (safe, node) == (False, "VIOLENT_REVENGE")


def test_triage_breaks_equal_severity_by_segment_order():
    """VIOLENT_REVENGE and TRAUMA_REPLAY share priority 1; the earlier one wins."""
    assert triage(["TRAUMA_REPLAY", "VIOLENT_REVENGE"], False)[1] == "TRAUMA_REPLAY"
    assert triage(["VIOLENT_REVENGE", "TRAUMA_REPLAY"], False)[1] == "VIOLENT_REVENGE"


def test_triage_fails_closed_on_unclassified():
    assert triage(["BEHAVIORAL_MASTERY"], True) == (False, "UNCLASSIFIED")


def test_triage_safe_case_reports_first_adaptive_segment():
    safe, node = triage(["AFFECT_EXPRESSION", "SOCIAL_MASTERY", "EMOTIONAL_MASTERY"], False)
    assert (safe, node) == (True, "SOCIAL_MASTERY")
    # With nothing adaptive, the first neutral segment names the case.
    assert triage(["NARRATIVE_SETTING", "AFFECT_EXPRESSION"], False) == (True, "NARRATIVE_SETTING")


def test_parse_segments_reads_type_and_node():
    cell = '[META] "I feel silly doing this..." → ? (?) | [ACTION] ctx:being chased "I turn..." → BEHAVIORAL_MASTERY (98%)'
    assert parse_segments(cell) == [("META", "?"), ("ACTION", "BEHAVIORAL_MASTERY")]


def test_parse_segments_tolerates_topup_format():
    """topup_s1_consistency.py omits the node suffix; the type must still parse."""
    assert parse_segments("[THOUGHT] It happens the same way || [ACTION] I end up trapped") == [
        ("THOUGHT", None),
        ("ACTION", None),
    ]


@pytest.mark.parametrize("system", sorted(CONSISTENCY_CSVS))
def test_triage_reimplementation_reproduces_pipeline_verdicts(system):
    """Pipeline segment labels through triage() must give the pipeline's verdict."""
    rows = load(CONSISTENCY_CSVS[system])
    reproduced, checked, _skipped = verify_pipeline_triage(rows)
    assert checked > 400
    assert reproduced == checked


@pytest.mark.parametrize("system", sorted(CONSISTENCY_CSVS))
def test_vote_derivation_round_trips_stored_consistency_fields(system):
    """Reconstructed votes must reproduce agreement_ratio and consistency_node."""
    _rows, diagnostics = derive_vote_verdicts(load(CONSISTENCY_CSVS[system]))
    assert diagnostics["roundtrip_checked"] == 477
    assert diagnostics["roundtrip_matches"] == diagnostics["roundtrip_checked"]


@pytest.mark.parametrize("system", sorted(CONSISTENCY_CSVS))
def test_only_meta_only_rows_fall_back_to_the_pipeline_verdict(system):
    rows, diagnostics = derive_vote_verdicts(load(CONSISTENCY_CSVS[system]))
    assert diagnostics["pipeline_fallback_rows"] == 3
    fallbacks = [row for row in rows if not row.get("vote_distribution")]
    assert all(row["pred_node"] == "META_ONLY" for row in fallbacks)


@pytest.mark.parametrize("system", sorted(CONSISTENCY_CSVS))
def test_every_scored_row_gets_a_vote_verdict(system):
    rows, _diagnostics = derive_vote_verdicts(load(CONSISTENCY_CSVS[system]))
    scored = [row for row in rows if row["gt_safe"] in ("True", "False")]
    assert len(scored) == 420
    assert all(row["vote_safe"] in ("True", "False") for row in scored)
    assert all(row["vote_node"] for row in scored)
