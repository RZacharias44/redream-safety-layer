"""Check how often the S2 rationale names the final classification code.

Backs the §4.7.1 sentence that the scored classification tokens mostly
restate a decision already written in the rationale. Mirrors
s2_logprob_diagnostic.py (same deterministic stratified sample of the dev
dataset, same classifier + critic path), but persists the parsed
context_summary / clinical_rationale fields so they can be inspected.

Fresh temperature-0 calls; descriptive check only. Result 2026-09-04
(50 cases, 116 calls, 114 parsed with a code):
    rationale names the CHOSEN code: 112/114
    rationale names ANY code:        113/114
    context_summary names chosen:      0/114
Output: data/benchmarks/dev_s2_rationale_check.csv

Usage:
    uv run python experiments/s2_rationale_check.py
"""
import asyncio
import csv
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

from s2_logprob_diagnostic import stratified_sample  # noqa: E402
from safety.classifier_cot import ClinicalClassifierCoT  # noqa: E402
from safety.critic import SafetyCritic  # noqa: E402

CODES = ["HIDE", "ESCAPE", "DESTROY", "DENY", "REPLAY", "CONFRONT",
         "HELP", "CHANGE", "RELAX", "POWER", "LOOK", "FEEL"]

OUT = REPO / "data/benchmarks/dev_s2_rationale_check.csv"


def parse_fields(raw: str) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return {"context_summary": "", "clinical_rationale": "", "parse_ok": False}
    return {
        "context_summary": str(data.get("context_summary", "")),
        "clinical_rationale": str(data.get("clinical_rationale", "")),
        "parse_ok": True,
    }


def has_code(text: str, code: str) -> bool:
    return re.search(rf"\b{code}\b", text) is not None


async def main() -> None:
    with open(REPO / "data/benchmarks/dev_dataset.json", encoding="utf-8") as f:
        suites = json.load(f)
    cases = stratified_sample(suites, 50)
    print(f"Sampled {len(cases)} cases")

    classifier = ClinicalClassifierCoT()
    classifier.logprob_diagnostics = []
    critic = SafetyCritic(classifier=classifier)

    rows = []
    for i, case in enumerate(cases, 1):
        before = len(classifier.logprob_diagnostics)
        try:
            await critic.evaluate_intervention(case["input"], nightmare_context=case["nightmare"])
            error = ""
        except Exception as e:  # keep going; a dropped case is not a missing answer
            error = str(e)
        for call in classifier.logprob_diagnostics[before:]:
            fields = parse_fields(call.get("raw_response", "") or "")
            code = call.get("code") or ""
            rows.append({
                "suite_id": case["suite_id"],
                "test": case["description"],
                "code": code,
                "parse_ok": fields["parse_ok"],
                "rationale_names_code": bool(code) and has_code(fields["clinical_rationale"], code),
                "summary_names_code": bool(code) and has_code(fields["context_summary"], code),
                "rationale_names_any_code": any(has_code(fields["clinical_rationale"], c) for c in CODES),
                "context_summary": fields["context_summary"],
                "clinical_rationale": fields["clinical_rationale"],
                "error": error,
            })
        print(f"  [{i}/{len(cases)}] {case['description'][:50]}", flush=True)

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    ok = [r for r in rows if r["parse_ok"] and r["code"]]
    named = sum(r["rationale_names_code"] for r in ok)
    named_any = sum(r["rationale_names_any_code"] for r in ok)
    summ = sum(r["summary_names_code"] for r in ok)
    print(f"\ncalls: {n} (parsed with code: {len(ok)})")
    print(f"rationale names the CHOSEN code: {named}/{len(ok)}")
    print(f"rationale names ANY code:        {named_any}/{len(ok)}")
    print(f"context_summary names chosen:    {summ}/{len(ok)}")


if __name__ == "__main__":
    asyncio.run(main())
