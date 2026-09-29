"""Build the four-tier scenario set for the §4.5 system-level eval.

Reads:
  - data/benchmarks/test_dataset.json          (suites + tests, ground truth)
  - data/benchmarks/test_dataset_annotation_log.json   (Ramon-vetted labels: VR, boundary, affect)
  - data/benchmarks/dev_dataset_spot_check_round2.json     (round-2 stratified spot check, 30 cases)
  - data/benchmarks/test_s1_baseline.csv  (post-fix critic predictions)

Writes:
  - data/system_eval/maladaptive_scenarios.json

Tiers (locked, no post-hoc expansion):
  main_unsafe       : 20 (4 per unsafe node, vetted-unsafe + critic-correct;
                          VR stratified 2 person + 2 object)
  safe_regression   : 10 (2 per adaptive node {BEH/SOC/ENV/EM/MM},
                          vetted-safe + critic pred_safe=True)
  multipart         : up to 6 (test_type=multipart, gt_safe=False, critic-correct)
  robustness        : up to 9 (3 false_positive + 3 false_negative + 3 wrong_node)

Random seed fixed for reproducibility.
"""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED = 4729  # 2026-04-29 fixed for reproducibility (Phase 4 selection)

UNSAFE_NODES = ("AVOIDANCE", "INTERRUPTION", "SUPPRESSION", "TRAUMA_REPLAY", "VIOLENT_REVENGE")
ADAPTIVE_NODES = ("BEHAVIORAL_MASTERY", "SOCIAL_MASTERY", "ENVIRONMENTAL_MASTERY",
                  "EMOTIONAL_MASTERY", "MYTHICAL_MASTERY")

# VR person-targeting suite (per Ramon's annotation log key_findings).
VR_PERSON_THEMES = {"Surgery while conscious", "Undergoing surgery while fully conscious and aware"}

EXCLUDED_SUITE = "Being buried alive, running out of air"


def normalize(text: str) -> str:
    return " ".join((text or "").strip().split()).lower()


def load_test_dataset(path: Path):
    """Returns flat list of dataset rows.

    Each row: {suite_idx (1-based), suite, nightmare, test_idx, description, input,
               expected_safe, expected_node, test_type}
    """
    with open(path) as f:
        suites = json.load(f)
    rows = []
    for s_idx, suite in enumerate(suites, 1):
        suite_name = suite.get("suite", "")
        nightmare = suite.get("nightmare", "")
        for t_idx, t in enumerate(suite.get("tests", []), 1):
            rows.append({
                "suite_idx": s_idx,
                "suite": suite_name,
                "nightmare": nightmare,
                "test_idx": t_idx,
                "description": t.get("description", ""),
                "input": t.get("input", ""),
                "expected_safe": t.get("expected_safe"),
                "expected_node": t.get("expected_node"),
                "test_type": t.get("test_type"),
            })
    return rows


def load_critic_preds(path: Path):
    """Returns dict keyed by normalized user_input → critic prediction row."""
    by_input = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            key = normalize(r.get("user_input", ""))
            if not key:
                continue
            by_input[key] = {
                "pred_safe": r.get("pred_safe") == "True",
                "pred_node": r.get("pred_node") or None,
                "conf": float(r["conf"]) if r.get("conf") not in (None, "", "None") else None,
                "severity": r.get("severity") or "",
            }
    return by_input


def load_ramon_vetted(annotation_path: Path, round2_path: Path):
    """Build dict normalized(input) → ramon_label dict."""
    vetted = {}

    def absorb(input_text, node, safe, source, theme=None):
        key = normalize(input_text)
        if not key:
            return
        # Prefer the first source we hit; treat duplicates as confirmation.
        if key in vetted:
            return
        vetted[key] = {
            "node": node,
            "safe": safe,
            "source": source,
            "theme": theme,
        }

    with open(annotation_path) as f:
        ann = json.load(f)

    for c in ann.get("boundary_cases", []):
        absorb(c.get("input", ""), c["ramon_label"]["node"], c["ramon_label"]["safe"],
               source="boundary", theme=c.get("theme"))
    for c in ann.get("affect_expression_cases", {}).get("cases", []):
        absorb(c.get("input", ""), c["ramon_label"]["node"], c["ramon_label"]["safe"],
               source="affect_expression", theme=c.get("theme"))
    for c in ann.get("violent_revenge_cases", {}).get("cases", []):
        absorb(c.get("input", ""), c["ramon_label"]["node"], c["ramon_label"]["safe"],
               source="violent_revenge_targeted", theme=c.get("theme"))

    with open(round2_path) as f:
        r2 = json.load(f)
    for c in r2:
        absorb(c.get("input", ""), c["node"], c["safe"],
               source="spot_check_r2", theme=c.get("suite"))

    return vetted


def select_main_unsafe(dataset_rows, critic_preds, vetted, rng):
    """4 per unsafe node, vetted-unsafe + critic correctly classifies (pred_safe=False, pred_node=gt).
    Diversify by suite. VR stratified 2 person + 2 object.
    Falls back to GPT-5.4 GT-only if vetted pool < 4 (60/60 spot-check agreement justifies this).
    """
    # Index dataset rows by normalized input.
    rows_by_input = {normalize(r["input"]): r for r in dataset_rows}

    selections = []
    coverage_notes = {}

    for node in UNSAFE_NODES:
        target = 4
        # Build candidates: dataset rows with gt_safe=False, gt_node=node,
        # that are vetted-unsafe by Ramon AND critic-correct.
        ramon_unsafe_pool = []
        gpt_only_pool = []  # fallback: critic-correct, GPT-5.4 GT only (no Ramon vet)

        for r in dataset_rows:
            if r["suite"] == EXCLUDED_SUITE:
                continue
            if r["expected_safe"] is not False:
                continue
            if r["expected_node"] != node:
                continue
            key = normalize(r["input"])
            cp = critic_preds.get(key)
            if cp is None:
                continue
            critic_correct = (cp["pred_safe"] is False) and (cp["pred_node"] == node)
            if not critic_correct:
                continue
            v = vetted.get(key)
            if v is not None and v["safe"] is False and v["node"] == node:
                ramon_unsafe_pool.append({"row": r, "pred": cp, "vetted": v})
            else:
                gpt_only_pool.append({"row": r, "pred": cp, "vetted": None})

        # Stratify VR.
        if node == "VIOLENT_REVENGE":
            def is_person(theme_or_suite):
                t = (theme_or_suite or "").lower()
                return ("surgery" in t) or any(p.lower() in t for p in VR_PERSON_THEMES)

            person_pool = [c for c in ramon_unsafe_pool if is_person(c["row"]["suite"])]
            object_pool = [c for c in ramon_unsafe_pool if not is_person(c["row"]["suite"])]
            chosen_p = sample_diverse(person_pool, 2, rng)
            chosen_o = sample_diverse(object_pool, 2, rng)
            chosen = chosen_p + chosen_o

            # Backfill from gpt_only_pool if vetted underflows.
            shortage_p = 2 - len(chosen_p)
            shortage_o = 2 - len(chosen_o)
            if shortage_p > 0:
                gpt_p = [c for c in gpt_only_pool if is_person(c["row"]["suite"])]
                chosen += sample_diverse(gpt_p, shortage_p, rng)
            if shortage_o > 0:
                gpt_o = [c for c in gpt_only_pool if not is_person(c["row"]["suite"])]
                chosen += sample_diverse(gpt_o, shortage_o, rng)
            coverage_notes[node] = {
                "person_pool_vetted": len(person_pool),
                "object_pool_vetted": len(object_pool),
                "selected_person": sum(1 for c in chosen if is_person(c["row"]["suite"])),
                "selected_object": sum(1 for c in chosen if not is_person(c["row"]["suite"])),
            }
        else:
            chosen = sample_diverse(ramon_unsafe_pool, target, rng)
            shortage = target - len(chosen)
            if shortage > 0:
                # Backfill from gpt_only_pool, avoiding suite collisions where possible.
                used_suites = {c["row"]["suite"] for c in chosen}
                fallback = sample_diverse(
                    [c for c in gpt_only_pool if c["row"]["suite"] not in used_suites],
                    shortage, rng)
                if len(fallback) < shortage:
                    fallback += sample_diverse(
                        [c for c in gpt_only_pool if c not in fallback], shortage - len(fallback), rng)
                chosen += fallback
            coverage_notes[node] = {
                "vetted_pool": len(ramon_unsafe_pool),
                "gpt_only_pool": len(gpt_only_pool),
                "selected_vetted": len([c for c in chosen if c["vetted"] is not None]),
                "selected_gpt_only": len([c for c in chosen if c["vetted"] is None]),
            }

        for i, c in enumerate(chosen, 1):
            sel = build_scenario(c, tier="main_unsafe", node=node, idx=i)
            if node == "VIOLENT_REVENGE":
                t = (c["row"]["suite"] or "").lower()
                sel["vr_subtype"] = "person_targeting" if "surgery" in t else "object_targeting"
            selections.append(sel)

    return selections, coverage_notes


def sample_diverse(pool, n, rng):
    """Pick up to n entries, preferring suite diversity."""
    if not pool or n <= 0:
        return []
    by_suite = defaultdict(list)
    for c in pool:
        by_suite[c["row"]["suite"]].append(c)
    suites = list(by_suite.keys())
    rng.shuffle(suites)
    chosen = []
    # Round-robin across suites first.
    while suites and len(chosen) < n:
        next_suites = []
        for s in suites:
            if not by_suite[s]:
                continue
            pick = rng.choice(by_suite[s])
            by_suite[s].remove(pick)
            chosen.append(pick)
            if len(chosen) >= n:
                break
            if by_suite[s]:
                next_suites.append(s)
        suites = next_suites
    return chosen[:n]


def select_safe_regression(dataset_rows, critic_preds, vetted, rng):
    """2 per adaptive node, vetted-safe + critic pred_safe=True."""
    selections = []
    coverage_notes = {}
    for node in ADAPTIVE_NODES:
        ramon_safe_pool = []
        gpt_only_pool = []
        for r in dataset_rows:
            if r["suite"] == EXCLUDED_SUITE:
                continue
            if r["expected_safe"] is not True:
                continue
            if r["expected_node"] != node:
                continue
            # Skip multipart for the safe_regression tier — it's about clean
            # safe inputs. (Multipart-safe doesn't really exist in this
            # dataset anyway, but be explicit.)
            if r["test_type"] != "clean":
                continue
            key = normalize(r["input"])
            cp = critic_preds.get(key)
            if cp is None or cp["pred_safe"] is not True:
                continue
            v = vetted.get(key)
            if v is not None and v["safe"] is True and v["node"] == node:
                ramon_safe_pool.append({"row": r, "pred": cp, "vetted": v})
            else:
                gpt_only_pool.append({"row": r, "pred": cp, "vetted": None})

        chosen = sample_diverse(ramon_safe_pool, 2, rng)
        shortage = 2 - len(chosen)
        if shortage > 0:
            used_suites = {c["row"]["suite"] for c in chosen}
            fallback = sample_diverse(
                [c for c in gpt_only_pool if c["row"]["suite"] not in used_suites],
                shortage, rng)
            if len(fallback) < shortage:
                fallback += sample_diverse(
                    [c for c in gpt_only_pool if c not in fallback],
                    shortage - len(fallback), rng)
            chosen += fallback

        coverage_notes[node] = {
            "vetted_pool": len(ramon_safe_pool),
            "gpt_only_pool": len(gpt_only_pool),
            "selected_vetted": len([c for c in chosen if c["vetted"] is not None]),
            "selected_gpt_only": len([c for c in chosen if c["vetted"] is None]),
        }
        for i, c in enumerate(chosen, 1):
            selections.append(build_scenario(c, tier="safe_regression", node=node, idx=i))
    return selections, coverage_notes


def select_multipart(dataset_rows, critic_preds, vetted, rng):
    """Up to 6 cases: test_type=multipart, gt_safe=False, critic-correct."""
    candidates = []
    for r in dataset_rows:
        if r["suite"] == EXCLUDED_SUITE:
            continue
        if r["test_type"] != "multipart":
            continue
        if r["expected_safe"] is not False:
            continue
        node = r["expected_node"]
        key = normalize(r["input"])
        cp = critic_preds.get(key)
        if cp is None:
            continue
        if cp["pred_safe"] is not False or cp["pred_node"] != node:
            continue
        v = vetted.get(key)
        candidates.append({"row": r, "pred": cp, "vetted": v})

    # Diversify by node first, then by suite.
    by_node = defaultdict(list)
    for c in candidates:
        by_node[c["row"]["expected_node"]].append(c)
    nodes_order = list(by_node.keys())
    rng.shuffle(nodes_order)
    chosen = []
    target = 6
    while nodes_order and len(chosen) < target:
        next_nodes = []
        for n in nodes_order:
            if not by_node[n]:
                continue
            pick = sample_diverse(by_node[n], 1, rng)
            if not pick:
                continue
            chosen.append(pick[0])
            by_node[n] = [c for c in by_node[n] if c is not pick[0]]
            if len(chosen) >= target:
                break
            if by_node[n]:
                next_nodes.append(n)
        nodes_order = next_nodes

    selections = []
    for i, c in enumerate(chosen, 1):
        s = build_scenario(c, tier="multipart", node=c["row"]["expected_node"], idx=i)
        selections.append(s)
    coverage = {"candidates": len(candidates), "selected": len(chosen),
                "by_node": {n: sum(1 for c in chosen if c["row"]["expected_node"] == n)
                            for n in {c["row"]["expected_node"] for c in chosen}}}
    return selections, coverage


def select_robustness(dataset_rows, critic_preds, vetted, rng):
    """3 false_positive + 3 false_negative + 3 wrong_node, vetted-only (Ramon-labeled)."""
    fp_pool = []  # gt_safe=True, pred_safe=False
    fn_pool = []  # gt_safe=False, pred_safe=True
    wn_pool = []  # gt_safe=False AND pred_safe=False AND pred_node != gt_node

    for r in dataset_rows:
        if r["suite"] == EXCLUDED_SUITE:
            continue
        key = normalize(r["input"])
        cp = critic_preds.get(key)
        if cp is None:
            continue
        v = vetted.get(key)
        if v is None:
            continue  # robustness tier requires Ramon-vetted ground truth

        ramon_safe = v["safe"]
        ramon_node = v["node"]
        pred_safe = cp["pred_safe"]
        pred_node = cp["pred_node"]

        if ramon_safe is True and pred_safe is False:
            fp_pool.append({"row": r, "pred": cp, "vetted": v})
        elif ramon_safe is False and pred_safe is True:
            fn_pool.append({"row": r, "pred": cp, "vetted": v})
        elif ramon_safe is False and pred_safe is False and pred_node != ramon_node:
            wn_pool.append({"row": r, "pred": cp, "vetted": v})

    fp_chosen = sample_diverse(fp_pool, 3, rng)
    fn_chosen = sample_diverse(fn_pool, 3, rng)
    wn_chosen = sample_diverse(wn_pool, 3, rng)

    selections = []
    for label, group in (("false_positive", fp_chosen),
                         ("false_negative", fn_chosen),
                         ("wrong_node", wn_chosen)):
        for i, c in enumerate(group, 1):
            s = build_scenario(c, tier="robustness", node=c["vetted"]["node"], idx=i,
                               subtier=label)
            selections.append(s)

    coverage = {
        "false_positive": {"pool": len(fp_pool), "selected": len(fp_chosen)},
        "false_negative": {"pool": len(fn_pool), "selected": len(fn_chosen)},
        "wrong_node": {"pool": len(wn_pool), "selected": len(wn_chosen)},
    }
    return selections, coverage


TIER_PREFIX = {
    "main_unsafe": "mu",
    "safe_regression": "sr",
    "multipart": "mp",
    "robustness": "rb",
}


def build_scenario(c, tier, node, idx, subtier=None):
    r = c["row"]
    cp = c["pred"]
    v = c["vetted"]
    prefix = TIER_PREFIX.get(tier, tier[:3])
    case_id = f"{prefix}_{node.lower()}_{idx:02d}"
    if subtier:
        sub_short = {"false_positive": "fp", "false_negative": "fn", "wrong_node": "wn"}.get(subtier, subtier[:2])
        case_id = f"{prefix}_{sub_short}_{node.lower()}_{idx:02d}"
    sc = {
        "case_id": case_id,
        "tier": tier,
        "expected_node": node,
        "nightmare": r["nightmare"],
        "user_input": r["input"],
        "source_suite": r["suite"],
        "source_test_description": r["description"],
        "ramon_label": ({"safe": v["safe"], "node": v["node"], "source": v["source"]}
                        if v else {"safe": r["expected_safe"], "node": r["expected_node"],
                                   "source": "gpt5.4_gt_only"}),
        "critic_pred": {"safe": cp["pred_safe"], "node": cp["pred_node"],
                        "conf": cp["conf"], "severity": cp["severity"]},
    }
    if subtier:
        sc["subtier"] = subtier
    return sc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/benchmarks/test_dataset.json")
    parser.add_argument("--annotation", default="data/benchmarks/test_dataset_annotation_log.json")
    parser.add_argument("--round2", default="data/benchmarks/dev_dataset_spot_check_round2.json")
    parser.add_argument("--critic-csv", default="data/benchmarks/test_s1_baseline.csv")
    parser.add_argument("--output", default="data/system_eval/maladaptive_scenarios.json")
    parser.add_argument("--coverage-out", default="data/system_eval/selection_coverage.json")
    args = parser.parse_args()

    rng = random.Random(SEED)

    dataset_rows = load_test_dataset(REPO_ROOT / args.dataset)
    critic_preds = load_critic_preds(REPO_ROOT / args.critic_csv)
    vetted = load_ramon_vetted(REPO_ROOT / args.annotation, REPO_ROOT / args.round2)

    print(f"Loaded {len(dataset_rows)} dataset rows")
    print(f"Loaded {len(critic_preds)} critic predictions")
    print(f"Loaded {len(vetted)} Ramon-vetted labels")

    main_unsafe, mu_cov = select_main_unsafe(dataset_rows, critic_preds, vetted, rng)
    safe_regression, sr_cov = select_safe_regression(dataset_rows, critic_preds, vetted, rng)
    multipart, mp_cov = select_multipart(dataset_rows, critic_preds, vetted, rng)
    robustness, rb_cov = select_robustness(dataset_rows, critic_preds, vetted, rng)

    all_scenarios = main_unsafe + safe_regression + multipart + robustness

    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_scenarios, f, indent=2, ensure_ascii=False)

    coverage = {
        "seed": SEED,
        "tiers": {
            "main_unsafe": {"selected": len(main_unsafe), "per_node": mu_cov},
            "safe_regression": {"selected": len(safe_regression), "per_node": sr_cov},
            "multipart": {"selected": len(multipart), "details": mp_cov},
            "robustness": {"selected": len(robustness), "details": rb_cov},
        },
        "totals": {
            "scenarios": len(all_scenarios),
            "main_unsafe": len(main_unsafe),
            "safe_regression": len(safe_regression),
            "multipart": len(multipart),
            "robustness": len(robustness),
        },
    }
    cov_path = REPO_ROOT / args.coverage_out
    with open(cov_path, "w") as f:
        json.dump(coverage, f, indent=2)

    print()
    print(f"Wrote {output_path} ({len(all_scenarios)} scenarios)")
    print(f"  main_unsafe: {len(main_unsafe)}")
    print(f"  safe_regression: {len(safe_regression)}")
    print(f"  multipart: {len(multipart)}")
    print(f"  robustness: {len(robustness)}")
    print(f"Coverage report: {cov_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
