"""Structural smoke test for SafetyCritic._build_constraint_message.

Constraint synthesis is pure string formatting from graph attributes (no LLM
calls). These tests are a regression guard for future refactors of
`graph_definitions.py` or the constraint template: they assert the returned
string has the expected sections, cites the node's clinical description and
at least one correction-edge target, and contains no unresolved `{...}`
placeholders.

Run with: uv run pytest tests/test_constraint_synthesis.py -v
"""

import os
import re
import pytest
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from safety.critic import SafetyCritic, SegmentResult
from safety.graph_definitions import build_clinical_graph, get_correction_strategies


UNSAFE_NODES = ["AVOIDANCE", "INTERRUPTION", "VIOLENT_REVENGE", "SUPPRESSION", "TRAUMA_REPLAY"]


@pytest.fixture(scope="module")
def critic():
    # SafetyCritic builds its Agents at init and needs a key string to exist;
    # these tests never call a model, so a placeholder is enough when no key
    # is configured. Scoped to this fixture so live tests still skip cleanly.
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("SCALEWAY_API_KEY", os.getenv("SCALEWAY_API_KEY") or "pytest-placeholder")
        yield SafetyCritic()


@pytest.fixture(scope="module")
def graph():
    return build_clinical_graph()


@pytest.mark.parametrize("node_id", UNSAFE_NODES)
def test_constraint_synthesis_structure(critic, graph, node_id):
    """For each unsafe node, _build_constraint_message returns a constraint with
    the expected header, sections, clinical description, and a correction target."""
    segment = SegmentResult(
        text=f"<fake segment text for {node_id}>",
        segment_type="ACTION",
        node_id=node_id,
        node_type="maladaptive",
        category=graph.nodes[node_id].get("category", "unknown"),
        confidence=0.9,
    )
    # Include a synthetic adaptive segment so the VALIDATE section is exercised
    # (the constraint template omits VALIDATE when adaptive_segments is empty).
    adaptive_segment = SegmentResult(
        text="<fake adaptive segment>",
        segment_type="ACTION",
        node_id="BEHAVIORAL_MASTERY",
        node_type="adaptive",
        category="mastery",
        confidence=0.9,
    )
    node_attrs = dict(graph.nodes[node_id])
    strategies = get_correction_strategies(graph, node_id)

    constraint = critic._build_constraint_message(
        primary_risk=segment,
        node_attrs=node_attrs,
        strategies=strategies,
        adaptive_segments=[adaptive_segment],
        unclassified_segments=None,
    )

    # Header
    assert constraint.startswith("[SAFETY SYSTEM - INTERVENTION REQUIRED]"), \
        f"{node_id}: missing safety-system header"

    # Clinical description of the node (first 40 chars should be unique enough)
    description = node_attrs.get("description", "")
    assert description, f"{node_id}: node has no description to cite"
    assert description[:40] in constraint, \
        f"{node_id}: clinical description not present in constraint"

    # At least one correction-edge target node name present
    assert strategies, f"{node_id}: no correction strategies defined in graph"
    assert any(s["target_node"] in constraint for s in strategies), \
        f"{node_id}: no correction-edge target node cited in constraint"

    # Section markers
    for marker in ("VALIDATE", "CORRECT", "INSTRUCTION", "IMPORTANT"):
        assert marker in constraint, f"{node_id}: missing '{marker}' section marker"

    # No unresolved template placeholders like {foo}
    unresolved = re.findall(r"\{[A-Za-z_][A-Za-z0-9_]*\}", constraint)
    assert not unresolved, f"{node_id}: unresolved placeholders {unresolved}"


# --- Tweak A coverage: empty-VALIDATE suppression for HIGH/CRITICAL ----------

HIGH_OR_CRITICAL_NODES = ["AVOIDANCE", "INTERRUPTION", "VIOLENT_REVENGE", "SUPPRESSION", "TRAUMA_REPLAY"]


@pytest.mark.parametrize("node_id", HIGH_OR_CRITICAL_NODES)
def test_no_validate_block_when_empty_adaptive_high_or_critical(critic, graph, node_id):
    """For HIGH/CRITICAL nodes with no adaptive segments, the constraint must:
    - emit a NO_VALIDATE block instead of an empty VALIDATE block
    - explicitly forbid opening with affirmation/praise/thanks
    - NOT contain the VALIDATE: header (so the LLM can't try to fill it)
    """
    segment = SegmentResult(
        text=f"<fake maladaptive segment for {node_id}>",
        segment_type="ACTION",
        node_id=node_id,
        node_type="maladaptive",
        category=graph.nodes[node_id].get("category", "unknown"),
        confidence=0.9,
    )
    node_attrs = dict(graph.nodes[node_id])
    strategies = get_correction_strategies(graph, node_id)

    constraint = critic._build_constraint_message(
        primary_risk=segment,
        node_attrs=node_attrs,
        strategies=strategies,
        adaptive_segments=[],
        unclassified_segments=None,
    )

    assert "NO_VALIDATE" in constraint, f"{node_id}: NO_VALIDATE block missing on empty adaptive + HIGH/CRITICAL"
    # Must NOT use the VALIDATE: header at the start of a line (which would invite
    # the LLM to validate). Check via regex so we don't false-match NO_VALIDATE:.
    assert not re.search(r"(?m)^VALIDATE:", constraint), \
        f"{node_id}: VALIDATE: header present despite empty adaptive segments"
    # Anti-affirmation language must be present
    for forbidden_phrase in ("affirmation", "praise"):
        assert forbidden_phrase in constraint.lower(), \
            f"{node_id}: NO_VALIDATE block missing '{forbidden_phrase}'"


def test_important_bullet_blocks_action_reframings(critic, graph):
    """The IMPORTANT bullet about 'Do NOT validate' must explicitly forbid
    reframing the maladaptive action as 'taking action' / 'being strong' / etc.
    This catches the VR_04 failure where the LLM said 'I appreciate that you're
    taking action and not giving up' about a violent revenge fantasy.
    """
    segment = SegmentResult(
        text="<violent revenge segment>",
        segment_type="ACTION",
        node_id="VIOLENT_REVENGE",
        node_type="maladaptive",
        category="violence",
        confidence=0.9,
    )
    node_attrs = dict(graph.nodes["VIOLENT_REVENGE"])
    strategies = get_correction_strategies(graph, "VIOLENT_REVENGE")
    constraint = critic._build_constraint_message(
        primary_risk=segment,
        node_attrs=node_attrs,
        strategies=strategies,
        adaptive_segments=[],
        unclassified_segments=None,
    )
    assert "taking action" in constraint, "IMPORTANT bullet must explicitly call out 'taking action' reframings"
