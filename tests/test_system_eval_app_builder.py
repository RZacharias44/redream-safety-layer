import json

from experiments.integration_eval.build_system_eval_app import (
    BLINDED_COLUMNS,
    REPO_ROOT,
    build_outputs,
    make_blinded_rows,
)
from experiments.integration_eval.export_for_experts import (
    RATER_COLUMNS,
    load_eval_rows,
    shuffle_with_key,
)


RESULTS = REPO_ROOT / "data/system_eval/system_eval_results.csv"


def test_system_eval_blinded_rows_exclude_condition_and_internal_fields():
    rows = load_eval_rows(RESULTS)
    shuffled, key = shuffle_with_key(rows, seed=42)
    blinded = make_blinded_rows(shuffled)
    payload = json.dumps(blinded)

    assert len(blinded) == 86
    assert set(blinded[0]) == set(BLINDED_COLUMNS)
    assert "condition" not in payload
    assert "tier" not in payload
    assert "expected_node" not in payload
    assert "detected_node" not in payload
    assert "constraint" not in payload
    assert {item["condition"] for item in key} == {"control", "treatment"}


def test_system_eval_rater_columns_match_revised_rubric():
    assert RATER_COLUMNS == [
        "shuffled_row_id",
        "nightmare",
        "user_input",
        "response",
        "expert_clinically_acceptable",
        "expert_redirect_quality",
        "expert_maladaptive_endorsement",
        "expert_validates_distress_appropriately",
        "expert_notes",
    ]


def test_build_system_eval_outputs_writes_static_materials(tmp_path):
    outputs = build_outputs(input_path=RESULTS, output_dir=tmp_path, seed=42)

    for path in outputs.values():
        assert path.exists()

    html = outputs["html"].read_text(encoding="utf-8")
    blinded = json.loads(outputs["blinded"].read_text(encoding="utf-8"))
    key = json.loads(outputs["key_json"].read_text(encoding="utf-8"))

    assert "ReDream System Eval" in html
    assert "Maladaptive endorsement" in html
    assert "redream_system_eval_${rater}_${stamp}.${extension}" in html
    assert len(blinded["cases"]) == 86
    assert len(key["key"]) == 86
    assert "condition" in key["key"][0]
    assert "condition" not in json.dumps(blinded["cases"])
