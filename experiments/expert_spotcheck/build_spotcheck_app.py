"""Build rater-blinded expert spot-check materials.

The generated HTML is a static local app. It embeds only blinded case content
and stores labels in browser localStorage until the rater exports JSON/CSV.

Usage:
    uv run python experiments/expert_spotcheck/build_spotcheck_app.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
EXCLUDED_SUITE = "Being buried alive, running out of air"

MALADAPTIVE_NODES = [
    "AVOIDANCE",
    "INTERRUPTION",
    "VIOLENT_REVENGE",
    "SUPPRESSION",
    "TRAUMA_REPLAY",
]
ADAPTIVE_NODES = [
    "BEHAVIORAL_MASTERY",
    "SOCIAL_MASTERY",
    "ENVIRONMENTAL_MASTERY",
    "EMOTIONAL_MASTERY",
    "MYTHICAL_MASTERY",
]
NEUTRAL_NODES = [
    "NARRATIVE_SETTING",
    "AFFECT_EXPRESSION",
]
NODE_ORDER = [
    "AVOIDANCE",
    "BEHAVIORAL_MASTERY",
    "INTERRUPTION",
    "SOCIAL_MASTERY",
    "VIOLENT_REVENGE",
    "ENVIRONMENTAL_MASTERY",
    "SUPPRESSION",
    "EMOTIONAL_MASTERY",
    "TRAUMA_REPLAY",
    "MYTHICAL_MASTERY",
    "NARRATIVE_SETTING",
    "AFFECT_EXPRESSION",
]
NODE_SAFE = {node: False for node in MALADAPTIVE_NODES}
NODE_SAFE.update({node: True for node in ADAPTIVE_NODES + NEUTRAL_NODES})


@dataclass(frozen=True)
class SourceCase:
    case_id: str
    suite: str
    nightmare: str
    user_input: str
    test_type: str
    source_description: str
    reference_safe: bool | None
    reference_node: str | None
    source_bucket: str


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def normalize_text(value: str) -> str:
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return " ".join(value.split())


def annotation_lookup(annotation_log: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup = {}
    for section_name in ("boundary_cases",):
        for item in annotation_log.get(section_name, []):
            label = item.get("ramon_label", {})
            lookup[normalize_text(item["input"])] = {
                "safe": label.get("safe"),
                "node": label.get("node"),
                "note": label.get("note", ""),
            }

    for section_name in ("affect_expression_cases", "violent_revenge_cases"):
        section = annotation_log.get(section_name, {})
        for item in section.get("cases", []):
            label = item.get("ramon_label", {})
            lookup[normalize_text(item["input"])] = {
                "safe": label.get("safe"),
                "node": label.get("node"),
                "note": label.get("note", ""),
            }
    return lookup


def load_source_cases(dataset_path: Path, annotation_path: Path) -> list[SourceCase]:
    dataset = load_json(dataset_path)
    annotations = annotation_lookup(load_json(annotation_path))
    cases: list[SourceCase] = []

    for suite_idx, suite in enumerate(dataset, start=1):
        suite_name = suite["suite"]
        if suite_name == EXCLUDED_SUITE:
            continue
        for test_idx, test in enumerate(suite["tests"], start=1):
            user_input = test["input"]
            test_type = test.get("test_type", "clean")
            anno = annotations.get(normalize_text(user_input), {})
            reference_node = test.get("expected_node")
            reference_safe = test.get("expected_safe")
            if test_type == "boundary":
                # Boundary cases have no generated labels, so their
                # reference_* columns carry the author's internal-review
                # annotation. They are neither benchmark reference labels nor
                # (on every row) the classifier's prediction; agreement
                # against them is not the classifier--expert comparison
                # reported in the thesis (see section_4_supplementary_claims).
                reference_node = anno.get("node")
                reference_safe = anno.get("safe")

            if reference_node is None and reference_safe is None:
                continue

            if test_type == "boundary":
                bucket = "boundary"
            elif reference_node == "AFFECT_EXPRESSION":
                bucket = "affect_expression"
            elif reference_node == "VIOLENT_REVENGE":
                bucket = "violent_revenge"
            else:
                bucket = "regular"

            cases.append(
                SourceCase(
                    case_id=f"ts{suite_idx:02d}_c{test_idx:02d}",
                    suite=suite_name,
                    nightmare=suite["nightmare"],
                    user_input=user_input,
                    test_type=test_type,
                    source_description=test.get("description", ""),
                    reference_safe=reference_safe,
                    reference_node=reference_node,
                    source_bucket=bucket,
                )
            )
    return cases


def shuffled(items: list[SourceCase], rng: random.Random) -> list[SourceCase]:
    copy = list(items)
    rng.shuffle(copy)
    return copy


def build_balanced_queue(
    cases: list[SourceCase],
    total: int,
    seed: int,
    block_size: int = 6,
) -> list[SourceCase]:
    """Build a prefix-valid order.

    Cases are grouped into same-nightmare blocks to reduce rereading burden.
    Each block reserves one boundary case when available and fills the remaining
    slots by round-robin over the ontology nodes. The first 36-case prefix is
    one six-case block per valid nightmare.
    """
    rng = random.Random(seed)
    suite_order = list(dict.fromkeys(case.suite for case in cases))

    boundaries_by_suite: dict[str, deque[SourceCase]] = {}
    by_suite_node: dict[str, dict[str, deque[SourceCase]]] = {}
    for suite in suite_order:
        boundaries_by_suite[suite] = deque(
            shuffled([c for c in cases if c.suite == suite and c.source_bucket == "boundary"], rng)
        )
        by_suite_node[suite] = {}
        for node in NODE_ORDER:
            node_cases = [
                c
                for c in cases
                if c.suite == suite and c.source_bucket != "boundary" and c.reference_node == node
            ]
            by_suite_node[suite][node] = deque(shuffled(node_cases, rng))

    queue: list[SourceCase] = []
    node_index = 0
    suite_index = 0

    while len(queue) < total:
        before = len(queue)
        suite = suite_order[suite_index % len(suite_order)]
        suite_index += 1
        block: list[SourceCase] = []

        if boundaries_by_suite[suite]:
            block.append(boundaries_by_suite[suite].popleft())

        while len(block) < block_size and len(queue) + len(block) < total:
            picked = None
            for _attempt in range(len(NODE_ORDER)):
                node = NODE_ORDER[node_index % len(NODE_ORDER)]
                node_index += 1
                if by_suite_node[suite][node]:
                    picked = by_suite_node[suite][node].popleft()
                    break
            if picked is None:
                for node in NODE_ORDER:
                    if by_suite_node[suite][node]:
                        picked = by_suite_node[suite][node].popleft()
                        break
            if picked is None:
                break
            block.append(picked)

        queue.extend(block[: total - len(queue)])
        if len(queue) == before:
            break

    if len(queue) < total:
        raise ValueError(f"Only built {len(queue)} cases, requested {total}.")
    return queue


def make_blinded_rows(queue: list[SourceCase]) -> list[dict[str, Any]]:
    rows = []
    for order, case in enumerate(queue, start=1):
        rows.append(
            {
                "order": order,
                "case_id": case.case_id,
                "nightmare": case.nightmare,
                "user_input": case.user_input,
            }
        )
    return rows


def make_key_rows(queue: list[SourceCase]) -> list[dict[str, Any]]:
    rows = []
    for order, case in enumerate(queue, start=1):
        rows.append(
            {
                "order": order,
                "case_id": case.case_id,
                "source_suite": case.suite,
                "source_test_description": case.source_description,
                "test_type": case.test_type,
                "source_bucket": case.source_bucket,
                "reference_safe": case.reference_safe,
                "reference_node": case.reference_node,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write empty CSV.")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_html(blinded_cases: list[dict[str, Any]], seed: int) -> str:
    data_json = json.dumps(
        {
            "metadata": {
                "dataset": "test_dataset.json",
                "seed": seed,
                "milestones": [36, 60, len(blinded_cases)],
                "node_groups": {
                    "maladaptive": MALADAPTIVE_NODES,
                    "adaptive": ADAPTIVE_NODES,
                    "neutral": NEUTRAL_NODES,
                },
            },
            "cases": blinded_cases,
        },
        ensure_ascii=False,
    )
    safe_data = data_json.replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ReDream Expert Spot Check</title>
  <style>
    :root {{
      --paper: #fbfaf7;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #66737f;
      --line: #d8ddd8;
      --sage: #586f61;
      --sage-soft: #edf3ee;
      --plum: #673d5f;
      --plum-soft: #f5edf3;
      --amber: #8a5d14;
      --amber-soft: #fff6df;
      --blue: #315f86;
      --blue-soft: #edf5fb;
      --danger: #a4382f;
      --safe: #39724a;
      --shadow: 0 12px 30px rgba(31, 41, 51, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
      overflow: hidden;
    }}
    button, input, textarea, select {{ font: inherit; }}
    .shell {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 390px;
      min-height: 100vh;
      height: 100vh;
      overflow: hidden;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 5;
      background: rgba(251, 250, 247, 0.94);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(10px);
    }}
    .topbar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 20px;
      padding: 18px 28px;
    }}
    h1 {{
      margin: 0;
      font-size: 20px;
      letter-spacing: 0;
      font-weight: 760;
    }}
    .meta {{
      display: flex;
      align-items: center;
      gap: 10px;
      color: var(--muted);
      font-size: 14px;
      white-space: nowrap;
    }}
    .pill {{
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 999px;
      padding: 5px 10px;
    }}
    main {{
      min-width: 0;
      height: 100vh;
      overflow-y: auto;
    }}
    .content {{
      max-width: 980px;
      margin: 0 auto;
      padding: 30px 28px 42px;
    }}
    .case-nav {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      margin-bottom: 22px;
    }}
    .nav-buttons {{
      display: flex;
      gap: 8px;
    }}
    .icon-button, .command-button {{
      min-height: 38px;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      border-radius: 6px;
      padding: 8px 12px;
      cursor: pointer;
    }}
    .icon-button:hover, .command-button:hover {{ border-color: var(--sage); }}
    .icon-button:focus-visible, .command-button:focus-visible, textarea:focus-visible, input:focus-visible {{
      outline: 3px solid rgba(49, 95, 134, 0.25);
      outline-offset: 2px;
    }}
    .case-number {{
      font-size: 14px;
      color: var(--muted);
      font-weight: 700;
    }}
    .text-block {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 22px;
      margin-bottom: 18px;
    }}
    .label {{
      margin: 0 0 10px;
      color: var(--sage);
      font-size: 13px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .nightmare {{
      font-family: Georgia, "Times New Roman", serif;
      font-size: 19px;
      line-height: 1.62;
    }}
    .attempt {{
      font-size: 22px;
      line-height: 1.5;
      font-weight: 650;
    }}
    aside {{
      border-left: 1px solid var(--line);
      background: #f4f2ed;
      padding: 24px;
      height: 100vh;
      overflow-y: auto;
    }}
    .panel {{
      display: grid;
      gap: 16px;
      padding-bottom: 24px;
    }}
    .field {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }}
    .field h2 {{
      margin: 0 0 12px;
      font-size: 15px;
    }}
    details.field {{
      padding: 0;
      overflow: hidden;
    }}
    details.field summary {{
      list-style: none;
      cursor: pointer;
      padding: 16px;
      font-size: 15px;
      font-weight: 760;
    }}
    details.field summary::-webkit-details-marker {{
      display: none;
    }}
    details.field summary::after {{
      content: "+";
      float: right;
      color: var(--muted);
      font-weight: 800;
    }}
    details.field[open] summary::after {{
      content: "-";
    }}
    .details-body {{
      border-top: 1px solid var(--line);
      padding: 14px 16px 16px;
    }}
    .instructions {{
      margin: 0;
      padding-left: 18px;
      color: var(--muted);
      font-size: 13px;
    }}
    .instructions li + li {{
      margin-top: 7px;
    }}
    .segmented {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }}
    .choice {{
      width: 100%;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 6px;
      padding: 10px;
      cursor: pointer;
      font-weight: 750;
    }}
    .choice[data-selected="true"][data-safe="true"] {{
      color: var(--safe);
      background: #edf8ef;
      border-color: var(--safe);
    }}
    .choice[data-selected="true"][data-safe="false"] {{
      color: var(--danger);
      background: #fff0ed;
      border-color: var(--danger);
    }}
    .choice[data-selected="true"][data-confidence] {{
      color: var(--blue);
      background: var(--blue-soft);
      border-color: var(--blue);
    }}
    .node-list {{
      display: grid;
      gap: 7px;
    }}
    .node-list + .node-list {{
      border-top: 1px solid var(--line);
      margin-top: 10px;
      padding-top: 10px;
    }}
    .node-option {{
      display: grid;
      grid-template-columns: 18px 1fr;
      gap: 8px;
      align-items: start;
      font-size: 13px;
      color: var(--ink);
    }}
    .node-option input {{ margin-top: 3px; }}
    .group-title {{
      margin: 0 0 6px;
      font-size: 12px;
      color: var(--muted);
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .confidence {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
    }}
    textarea {{
      width: 100%;
      min-height: 92px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      resize: vertical;
    }}
    .progress {{
      display: grid;
      gap: 8px;
    }}
    .bar {{
      height: 10px;
      border-radius: 999px;
      background: #e3e1dc;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, var(--sage), var(--blue));
    }}
    .milestones {{
      display: flex;
      justify-content: space-between;
      color: var(--muted);
      font-size: 12px;
    }}
    .exports {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }}
    .rater {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px;
    }}
    .status {{
      color: var(--muted);
      font-size: 13px;
      min-height: 20px;
    }}
    .guidance {{
      color: var(--muted);
      font-size: 13px;
      margin: 8px 0 0;
    }}
    .reference {{
      display: grid;
      gap: 10px;
    }}
    .reference-group {{
      border-top: 1px solid var(--line);
      padding-top: 10px;
    }}
    .reference-group:first-child {{
      border-top: 0;
      padding-top: 0;
    }}
    .reference-title {{
      margin: 0 0 7px;
      color: var(--sage);
      font-size: 12px;
      font-weight: 820;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .definition-list {{
      display: grid;
      gap: 6px;
      margin: 0;
      font-size: 12px;
      color: var(--muted);
    }}
    .definition-list div {{
      display: grid;
      grid-template-columns: 118px 1fr;
      gap: 8px;
      align-items: start;
    }}
    .definition-list dt {{
      color: var(--ink);
      font-weight: 760;
    }}
    .definition-list dd {{
      margin: 0;
    }}
    .boundary-rules {{
      margin: 0;
      padding-left: 18px;
      color: var(--muted);
      font-size: 12px;
    }}
    .boundary-rules li + li {{
      margin-top: 6px;
    }}
    @media (max-width: 980px) {{
      body {{ overflow: auto; }}
      .shell {{ grid-template-columns: 1fr; height: auto; overflow: visible; }}
      main {{ height: auto; overflow: visible; }}
      aside {{ border-left: 0; border-top: 1px solid var(--line); height: auto; overflow: visible; }}
      .panel {{ position: static; }}
      .topbar {{ align-items: flex-start; flex-direction: column; }}
      .meta {{ white-space: normal; flex-wrap: wrap; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <main>
      <header>
        <div class="topbar">
          <h1>ReDream Expert Spot Check</h1>
          <div class="meta">
            <span class="pill" id="answered-pill">0 rated</span>
            <span class="pill">36 minimum</span>
            <span class="pill">60 target</span>
            <span class="pill" id="total-pill"></span>
          </div>
        </div>
      </header>
      <div class="content">
        <div class="case-nav">
          <div>
            <div class="case-number" id="case-number"></div>
            <div class="status" id="case-status"></div>
          </div>
          <div class="nav-buttons">
            <button class="icon-button" id="prev" title="Previous case">←</button>
            <button class="icon-button" id="next" title="Next case">→</button>
          </div>
        </div>
        <section class="text-block">
          <p class="label">Nightmare context</p>
          <div class="nightmare" id="nightmare"></div>
        </section>
        <section class="text-block">
          <p class="label">Patient rescripting attempt</p>
          <div class="attempt" id="attempt"></div>
        </section>
      </div>
    </main>
    <aside>
      <div class="panel">
        <section class="field">
          <h2>Rater</h2>
          <input class="rater" id="rater-id" placeholder="Initials or rater ID">
        </section>
        <details class="field" open>
          <summary>Rating instructions</summary>
          <div class="details-body">
            <ul class="instructions">
              <li>Rate cases in order. Stopping at any point is fine; skipping around is not.</li>
              <li>First decide whether the rescripting attempt is safe or unsafe.</li>
              <li>Choose one primary clinical node: the pattern that should guide the therapeutic response or safety intervention.</li>
              <li>If safe and unsafe elements both appear, prioritize the unsafe element if it would require redirecting the patient.</li>
              <li>Use notes for secondary labels, ambiguity, low confidence, or low ecological validity.</li>
              <li>Export the file before closing the browser. Clear local labels before the next rater uses this laptop.</li>
            </ul>
          </div>
        </details>
        <section class="field">
          <h2>Safety label</h2>
          <div class="segmented">
            <button class="choice" data-safe="true" id="safe-true">Safe</button>
            <button class="choice" data-safe="false" id="safe-false">Unsafe</button>
          </div>
        </section>
        <section class="field">
          <h2>Clinical node</h2>
          <p class="guidance">Choose the primary label. If more than one label seems plausible, use notes for secondary labels or ambiguity.</p>
          <div id="nodes"></div>
        </section>
        <details class="field reference">
          <summary>Ontology reference</summary>
          <div class="details-body">
            <div class="reference-group">
              <p class="reference-title">Maladaptive</p>
              <dl class="definition-list">
                <div><dt>Avoidance</dt><dd>Hiding, fleeing, freezing, doing nothing, or disappearing from the dream scene instead of changing it.</dd></div>
                <div><dt>Interruption</dt><dd>Ending the dream by waking up, forcing it to stop, or cutting the scene off.</dd></div>
                <div><dt>Violent revenge</dt><dd>Rage-driven, excessive, or retaliatory harm beyond proportionate self-protection.</dd></div>
                <div><dt>Suppression</dt><dd>Denying, minimizing, numbing, or pushing away emotion in a clearly distressing scene.</dd></div>
                <div><dt>Trauma replay</dt><dd>Repeating the nightmare or traumatic event unchanged, with no new mastery action.</dd></div>
              </dl>
            </div>
            <div class="reference-group">
              <p class="reference-title">Adaptive mastery</p>
              <dl class="definition-list">
                <div><dt>Behavioral</dt><dd>Active, goal-directed action: facing, stopping, escaping with agency, or proportionate self-defense.</dd></div>
                <div><dt>Social</dt><dd>Seeking, receiving, or creating support from people, animals, or helpful figures.</dd></div>
                <div><dt>Environmental</dt><dd>Changing the setting, objects, barriers, routes, light, distance, or dream rules.</dd></div>
                <div><dt>Emotional</dt><dd>Acknowledging and regulating emotion, choosing calm/courage, or feeling safe while staying engaged.</dd></div>
                <div><dt>Mythical</dt><dd>Using magical, symbolic, supernatural, or imaginative powers to transform the dream.</dd></div>
              </dl>
            </div>
            <div class="reference-group">
              <p class="reference-title">Neutral</p>
              <dl class="definition-list">
                <div><dt>Narrative setting</dt><dd>Scene description or factual dream detail without a safety-relevant action.</dd></div>
                <div><dt>Affect expression</dt><dd>Stating fear, sadness, relief, or other emotion without suppressing it or changing the dream.</dd></div>
              </dl>
            </div>
            <div class="reference-group">
              <p class="reference-title">Boundary rules</p>
              <ul class="boundary-rules">
                <li>If safe and unsafe material both appear, choose the unsafe node if a therapist would need to redirect it.</li>
                <li>Proportionate protection is behavioral mastery; revenge, punishment, or continued harm is violent revenge.</li>
                <li>Emotional mastery acknowledges emotion; suppression denies, minimizes, or numbs it.</li>
                <li>Leaving danger inside the dream can be behavioral mastery; waking up or ending the dream is interruption.</li>
              </ul>
            </div>
          </div>
        </details>
        <section class="field">
          <h2>Confidence</h2>
          <div class="confidence">
            <button class="choice" data-confidence="low">Low</button>
            <button class="choice" data-confidence="medium">Medium</button>
            <button class="choice" data-confidence="high">High</button>
          </div>
        </section>
        <section class="field">
          <h2>Notes</h2>
          <textarea id="notes"></textarea>
        </section>
        <section class="field progress">
          <div class="bar"><div class="bar-fill" id="bar-fill"></div></div>
          <div class="milestones"><span>0</span><span>36</span><span>60</span><span id="max-milestone"></span></div>
          <div class="exports">
            <button class="command-button" id="export-json">Export JSON</button>
            <button class="command-button" id="export-csv">Export CSV</button>
          </div>
          <button class="command-button" id="clear-labels">Clear local labels</button>
          <div class="status" id="save-status"></div>
        </section>
      </div>
    </aside>
  </div>
  <script id="case-data" type="application/json">{safe_data}</script>
  <script>
    const payload = JSON.parse(document.getElementById("case-data").textContent);
    const cases = payload.cases;
    const metadata = payload.metadata;
    const storageKey = "redream_expert_spotcheck_v1_seed_" + metadata.seed;
    const raterKey = storageKey + "_rater";
    let current = 0;
    let labels = JSON.parse(localStorage.getItem(storageKey) || "{{}}");

    const $ = (id) => document.getElementById(id);
    const nodeLabels = {{
      AVOIDANCE: "Avoidance",
      INTERRUPTION: "Interruption",
      VIOLENT_REVENGE: "Violent revenge",
      SUPPRESSION: "Suppression",
      TRAUMA_REPLAY: "Trauma replay",
      BEHAVIORAL_MASTERY: "Behavioral mastery",
      SOCIAL_MASTERY: "Social mastery",
      ENVIRONMENTAL_MASTERY: "Environmental mastery",
      EMOTIONAL_MASTERY: "Emotional mastery",
      MYTHICAL_MASTERY: "Mythical mastery",
      NARRATIVE_SETTING: "Narrative setting",
      AFFECT_EXPRESSION: "Affect expression"
    }};

    function escapeCsv(value) {{
      const text = String(value ?? "");
      return '"' + text.replaceAll('"', '""') + '"';
    }}

    function currentLabel() {{
      const id = cases[current].case_id;
      labels[id] ||= {{}};
      return labels[id];
    }}

    function save() {{
      localStorage.setItem(storageKey, JSON.stringify(labels));
      localStorage.setItem(raterKey, $("rater-id").value.trim());
      $("save-status").textContent = "Saved locally";
      updateProgress();
    }}

    function isComplete(label) {{
      return label && typeof label.safe === "boolean" && label.node;
    }}

    function updateProgress() {{
      const answered = cases.filter((c) => isComplete(labels[c.case_id])).length;
      $("answered-pill").textContent = `${{answered}} rated`;
      $("bar-fill").style.width = `${{Math.round((answered / cases.length) * 100)}}%`;
    }}

    function buildNodes() {{
      const root = $("nodes");
      root.innerHTML = "";
      const groups = [
        ["Maladaptive", metadata.node_groups.maladaptive],
        ["Adaptive", metadata.node_groups.adaptive],
        ["Neutral", metadata.node_groups.neutral],
      ];
      for (const [title, nodes] of groups) {{
        const wrap = document.createElement("div");
        wrap.className = "node-list";
        const heading = document.createElement("p");
        heading.className = "group-title";
        heading.textContent = title;
        wrap.appendChild(heading);
        for (const node of nodes) {{
          const label = document.createElement("label");
          label.className = "node-option";
          label.innerHTML = `<input type="radio" name="node" value="${{node}}"><span>${{nodeLabels[node]}}</span>`;
          wrap.appendChild(label);
        }}
        root.appendChild(wrap);
      }}
      root.addEventListener("change", (event) => {{
        if (event.target.name === "node") {{
          currentLabel().node = event.target.value;
          currentLabel().updated_at = new Date().toISOString();
          save();
          renderStatus();
        }}
      }});
    }}

    function renderStatus() {{
      const label = currentLabel();
      const labelState = isComplete(label) ? "Complete" : "Needs label";
      const contextState = current > 0 && cases[current].nightmare === cases[current - 1].nightmare
        ? "same nightmare as previous"
        : "new nightmare context";
      $("case-status").textContent = `${{labelState}} · ${{contextState}}`;
    }}

    function renderCase() {{
      const item = cases[current];
      const label = currentLabel();
      $("case-number").textContent = `Case ${{item.order}} of ${{cases.length}}`;
      $("nightmare").textContent = item.nightmare;
      $("attempt").textContent = item.user_input;
      $("prev").disabled = current === 0;
      $("next").disabled = current === cases.length - 1;
      $("safe-true").dataset.selected = label.safe === true ? "true" : "false";
      $("safe-false").dataset.selected = label.safe === false ? "true" : "false";
      document.querySelectorAll('input[name="node"]').forEach((input) => {{
        input.checked = input.value === label.node;
      }});
      document.querySelectorAll("[data-confidence]").forEach((button) => {{
        button.dataset.selected = button.dataset.confidence === label.confidence ? "true" : "false";
      }});
      $("notes").value = label.notes || "";
      renderStatus();
      updateProgress();
    }}

    function setSafe(value) {{
      currentLabel().safe = value;
      currentLabel().updated_at = new Date().toISOString();
      save();
      renderCase();
    }}

    function setConfidence(value) {{
      currentLabel().confidence = value;
      currentLabel().updated_at = new Date().toISOString();
      save();
      renderCase();
    }}

    function exportRows() {{
      const rater = $("rater-id").value.trim();
      return cases.map((item) => {{
        const label = labels[item.case_id] || {{}};
        return {{
          rater_id: rater,
          order: item.order,
          case_id: item.case_id,
          expert_safe: label.safe === undefined ? "" : String(label.safe),
          expert_node: label.node || "",
          expert_confidence: label.confidence || "",
          expert_notes: label.notes || "",
          updated_at: label.updated_at || "",
        }};
      }});
    }}

    function exportName(extension) {{
      const rater = $("rater-id").value.trim().replace(/[^a-zA-Z0-9_-]+/g, "_") || "unknown_rater";
      const stamp = new Date().toISOString().replace(/[:.]/g, "-");
      return `redream_spotcheck_${{rater}}_${{stamp}}.${{extension}}`;
    }}

    function download(name, content, type) {{
      const blob = new Blob([content], {{ type }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = name;
      a.click();
      URL.revokeObjectURL(url);
    }}

    function exportJson() {{
      const content = JSON.stringify({{
        metadata,
        exported_at: new Date().toISOString(),
        rows: exportRows(),
      }}, null, 2);
      download(exportName("json"), content, "application/json");
    }}

    function exportCsv() {{
      const rows = exportRows();
      const columns = Object.keys(rows[0]);
      const lines = [columns.map(escapeCsv).join(",")];
      for (const row of rows) {{
        lines.push(columns.map((col) => escapeCsv(row[col])).join(","));
      }}
      download(exportName("csv"), lines.join("\\n"), "text/csv");
    }}

    function clearLabels() {{
      const answered = cases.filter((c) => isComplete(labels[c.case_id])).length;
      if (answered > 0 && !window.confirm(`Clear ${{answered}} locally saved ratings from this browser? Export first if you need them.`)) {{
        return;
      }}
      labels = {{}};
      localStorage.removeItem(storageKey);
      localStorage.removeItem(raterKey);
      $("rater-id").value = "";
      $("save-status").textContent = "Local labels cleared";
      current = 0;
      renderCase();
    }}

    $("rater-id").value = localStorage.getItem(raterKey) || "";
    $("rater-id").addEventListener("input", save);
    $("safe-true").addEventListener("click", () => setSafe(true));
    $("safe-false").addEventListener("click", () => setSafe(false));
    $("prev").addEventListener("click", () => {{ current = Math.max(0, current - 1); renderCase(); }});
    $("next").addEventListener("click", () => {{ current = Math.min(cases.length - 1, current + 1); renderCase(); }});
    $("notes").addEventListener("input", () => {{
      currentLabel().notes = $("notes").value;
      currentLabel().updated_at = new Date().toISOString();
      save();
    }});
    document.querySelectorAll("[data-confidence]").forEach((button) => {{
      button.addEventListener("click", () => setConfidence(button.dataset.confidence));
    }});
    $("export-json").addEventListener("click", exportJson);
    $("export-csv").addEventListener("click", exportCsv);
    $("clear-labels").addEventListener("click", clearLabels);
    document.addEventListener("keydown", (event) => {{
      if (event.target.matches("textarea,input")) return;
      if (event.key === "ArrowLeft") {{ current = Math.max(0, current - 1); renderCase(); }}
      if (event.key === "ArrowRight") {{ current = Math.min(cases.length - 1, current + 1); renderCase(); }}
    }});

    $("total-pill").textContent = `${{cases.length}} extended`;
    $("max-milestone").textContent = String(cases.length);
    buildNodes();
    renderCase();
  </script>
</body>
</html>
"""


def build_outputs(
    dataset_path: Path,
    annotation_path: Path,
    output_dir: Path,
    seed: int,
    total: int,
) -> dict[str, Path]:
    source_cases = load_source_cases(dataset_path, annotation_path)
    queue = build_balanced_queue(source_cases, total=total, seed=seed)
    blinded = make_blinded_rows(queue)
    key = make_key_rows(queue)

    output_dir.mkdir(parents=True, exist_ok=True)
    blinded_path = output_dir / "spotcheck_cases_blinded.json"
    key_path = output_dir / "spotcheck_key.json"
    key_csv_path = output_dir / "spotcheck_key.csv"
    html_path = output_dir / "spotcheck_app.html"

    blinded_path.write_text(json.dumps({"seed": seed, "cases": blinded}, indent=2, ensure_ascii=False), encoding="utf-8")
    key_path.write_text(json.dumps({"seed": seed, "key": key}, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(key_csv_path, key)
    html_path.write_text(render_html(blinded, seed=seed), encoding="utf-8")

    return {
        "html": html_path,
        "blinded": blinded_path,
        "key_json": key_path,
        "key_csv": key_csv_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/benchmarks/test_dataset.json")
    parser.add_argument("--annotations", default="data/benchmarks/test_dataset_annotation_log.json")
    parser.add_argument("--output-dir", default="data/expert_spotcheck")
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--total", type=int, default=100)
    args = parser.parse_args()

    outputs = build_outputs(
        dataset_path=REPO_ROOT / args.dataset,
        annotation_path=REPO_ROOT / args.annotations,
        output_dir=REPO_ROOT / args.output_dir,
        seed=args.seed,
        total=args.total,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
