import json

from experiments.expert_spotcheck.build_spotcheck_app import (
    EXCLUDED_SUITE,
    NODE_ORDER,
    REPO_ROOT,
    build_balanced_queue,
    build_outputs,
    load_source_cases,
    make_blinded_rows,
)


DATASET = REPO_ROOT / "data/benchmarks/test_dataset.json"
ANNOTATIONS = REPO_ROOT / "data/benchmarks/test_dataset_annotation_log.json"


def test_balanced_prefix_contains_boundaries_and_all_nodes():
    cases = load_source_cases(DATASET, ANNOTATIONS)
    queue = build_balanced_queue(cases, total=100, seed=20260622)
    prefix = queue[:36]

    boundary_count = sum(case.source_bucket == "boundary" for case in prefix)
    nodes = {case.reference_node for case in prefix if case.source_bucket != "boundary"}
    blocks = [prefix[i : i + 6] for i in range(0, 36, 6)]

    assert len(queue) == 100
    assert boundary_count == 6
    assert nodes == set(NODE_ORDER)
    assert len({block[0].suite for block in blocks}) == 6
    assert all(len({case.suite for case in block}) == 1 for block in blocks)
    assert all(case.suite != EXCLUDED_SUITE for case in queue)


def test_blinded_rows_do_not_include_reference_labels():
    cases = load_source_cases(DATASET, ANNOTATIONS)
    queue = build_balanced_queue(cases, total=36, seed=20260622)
    blinded = make_blinded_rows(queue)

    assert set(blinded[0]) == {"order", "case_id", "nightmare", "user_input"}
    assert "reference_node" not in json.dumps(blinded)
    assert "reference_safe" not in json.dumps(blinded)


def test_build_outputs_writes_static_materials(tmp_path):
    outputs = build_outputs(
        dataset_path=DATASET,
        annotation_path=ANNOTATIONS,
        output_dir=tmp_path,
        seed=20260622,
        total=36,
    )

    for path in outputs.values():
        assert path.exists()

    html = outputs["html"].read_text(encoding="utf-8")
    key = json.loads(outputs["key_json"].read_text(encoding="utf-8"))

    assert "ReDream Expert Spot Check" in html
    assert "Clear local labels" in html
    assert "redream_spotcheck_${rater}_${stamp}.${extension}" in html
    assert len(key["key"]) == 36
    assert "reference_node" in key["key"][0]
