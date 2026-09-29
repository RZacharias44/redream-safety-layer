#!/usr/bin/env python3
"""Generate the appendix benchmark-inventory tables from the dataset files.

Reads data/benchmarks/{dev_dataset,test_dataset,multi_maladaptive_dataset}.json and
the BOUNDARY_PAIRS definition in experiments/generate_dataset.py (parsed via
ast, no import side effects), and writes
data/evaluation_statistics/chapter4/benchmark_inventory_tables.tex.

All counts are computed from the dataset files at generation time; nothing is
hard-coded. Rerun after any dataset change:

    uv run python experiments/evaluation_statistics/generate_benchmark_inventory.py
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / "data/benchmarks/dev_dataset.json"
TEST = ROOT / "data/benchmarks/test_dataset.json"
MAL_MAL = ROOT / "data/benchmarks/multi_maladaptive_dataset.json"
GEN_SCRIPT = ROOT / "experiments/generate_dataset.py"
OUT = ROOT / "data/evaluation_statistics/chapter4/benchmark_inventory_tables.tex"

# The held-out suite excluded from the final test set (truncated generation).
EXCLUDED_TEST_SUITE = "Being buried alive, running out of air"


def latex_escape(s: str) -> str:
    for a, b in [("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$")]:
        s = s.replace(a, b)
    return s


def load_suites(path: Path) -> list[tuple[str, int]]:
    data = json.loads(path.read_text())
    return [(s["suite"], len(s["tests"])) for s in data]


def load_boundary_pairs() -> list[dict]:
    """Parse BOUNDARY_PAIRS out of generate_dataset.py without importing it."""
    tree = ast.parse(GEN_SCRIPT.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "BOUNDARY_PAIRS":
                    return ast.literal_eval(node.value)
    raise SystemExit("BOUNDARY_PAIRS not found in generate_dataset.py")


def theme_table(dev, test, mal_suites) -> str:
    lines = []
    lines.append(r"\begingroup")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.16}")
    lines.append(
        r"\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.14\linewidth}"
        r">{\raggedright\arraybackslash}p{0.57\linewidth}"
        r">{\raggedleft\arraybackslash}p{0.11\linewidth}"
        r">{\raggedright\arraybackslash}p{0.10\linewidth}@{}}"
    )
    lines.append(
        r"  \caption[Complete inventory of benchmark nightmare scenarios.]"
        r"{Complete inventory of benchmark nightmare scenarios. "
        r"Scenarios marked with * also contribute seven cases each to the "
        r"supplementary multi-maladaptive slice "
        rf"({sum(c for _, c in mal_suites)} cases in total).}}"
    )
    lines.append(r"  \label{tab:app-theme-inventory}\\")
    header = r"  Dataset & Scenario theme & Cases & Note \\"
    lines += [r"  \toprule", header, r"  \midrule", r"  \endfirsthead",
              r"  \toprule", header, r"  \midrule", r"  \endhead"]
    mal_names = {n for n, _ in mal_suites}
    dev_total = sum(c for _, c in dev)
    for i, (name, cases) in enumerate(dev):
        label = "Development" if i == 0 else ""
        star = "*" if name in mal_names else ""
        lines.append(rf"  {label} & {latex_escape(name)}{star} & {cases} & \\")
    lines.append(rf"  & \emph{{Total}} & {dev_total} & \\")
    lines.append(r"  \midrule")
    test_total = sum(c for _, c in test)
    kept_total = sum(c for n, c in test if n != EXCLUDED_TEST_SUITE)
    for i, (name, cases) in enumerate(test):
        label = "Held-out" if i == 0 else ""
        note = "excluded" if name == EXCLUDED_TEST_SUITE else ""
        lines.append(rf"  {label} & {latex_escape(name)} & {cases} & {note} \\")
    lines.append(
        rf"  & \emph{{Total (generated / final)}} & {test_total} / {kept_total} & \\"
    )
    lines += [r"  \bottomrule", r"\end{longtable}", r"\endgroup"]
    return "\n".join(lines)


def boundary_table(pairs) -> str:
    lines = []
    lines.append(r"\begingroup")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.16}")
    lines.append(
        r"\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.30\linewidth}"
        r">{\raggedright\arraybackslash}p{0.19\linewidth}"
        r">{\raggedright\arraybackslash}p{0.43\linewidth}@{}}"
    )
    lines.append(
        r"  \caption[The six targeted decision boundaries used to generate "
        r"boundary cases.]{The six targeted decision boundaries used to "
        r"generate boundary cases, as defined in the generation script.}"
    )
    lines.append(r"  \label{tab:app-boundary-pairs}\\")
    header = r"  Category pair & Boundary & Description \\"
    lines += [r"  \toprule", header, r"  \midrule", r"  \endfirsthead",
              r"  \toprule", header, r"  \midrule", r"  \endhead"]
    for p in pairs:
        pair = (
            rf"\texttt{{{latex_escape(p['nodeA'])}}} vs.\ "
            rf"\texttt{{{latex_escape(p['nodeB'])}}}"
        )
        boundary = latex_escape(p["boundary"].replace("_", " "))
        desc = latex_escape(p["description"])
        lines.append(rf"  {pair} & {boundary} & {desc} \\")
    lines += [r"  \bottomrule", r"\end{longtable}", r"\endgroup"]
    return "\n".join(lines)


def main() -> None:
    dev = load_suites(DEV)
    test = load_suites(TEST)
    mal = load_suites(MAL_MAL)
    pairs = load_boundary_pairs()

    # Sanity checks against the numbers reported in Chapter 4.
    assert sum(c for _, c in dev) == 480, "dev total changed"
    assert sum(c for _, c in test) == 315, "test total changed"
    assert sum(c for n, c in test if n != EXCLUDED_TEST_SUITE) == 267, "final test total changed"
    assert sum(c for _, c in mal) == 35, "mal+mal total changed"
    assert {n for n, _ in mal} <= {n for n, _ in dev}, "mal+mal suite not in dev set"
    assert len(pairs) == 6, "boundary pair count changed"

    header = (
        "% Generated by experiments/evaluation_statistics/"
        "generate_benchmark_inventory.py; do not edit by hand.\n"
    )
    OUT.write_text(
        header + theme_table(dev, test, mal) + "\n\n" + boundary_table(pairs) + "\n"
    )
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
