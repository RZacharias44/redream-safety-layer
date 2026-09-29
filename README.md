# ReDream AI safety layer

A neuro-symbolic safety layer for an LLM-assisted imagery rehearsal therapy
(IRT) chatbot, together with the baseline chatbot, the evaluation data and the
scripts that compute every statistic reported in the thesis

> Ramon Yassin Zacharias. *A Neuro-Symbolic Safety Layer for LLM-Assisted
> Imagery Rehearsal Therapy.* Master's thesis, Osnabrück University, Institute
> of Cognitive Science, 2026.

> **Research prototype — not for clinical use.** This software was built and
> evaluated for a master's thesis. It is not a medical device, it has not been
> validated for clinical use, and it does not handle crisis situations. The
> maladaptive categories it enforces are a design position grounded in the IRT
> literature, not a clinical consensus. Anyone deploying an intervention such
> as IRT through an AI system is responsible for the applicable regulation.

## What the safety layer does

Imagery rehearsal therapy asks a patient to rewrite a recurrent nightmare and
rehearse the new version. When a chatbot guides that rewriting, the patient's
proposals can drift into patterns that the clinical literature treats as
counter-therapeutic: fleeing or hiding, waking up to escape, violent revenge,
suppressing the feeling, or replaying the trauma unchanged. The safety layer
sits between the user's rewriting proposal and the response generator:

1. **Segmenter** — an LLM call splits the user turn into spans typed as
   ACTION, THOUGHT, FEELING or META (conversation about the task).
2. **META filter** — conversational spans are set aside; only dream content is
   classified.
3. **Classifier** — each span is mapped to one node of a clinical ontology.
   Two variants exist: *System 1* emits a classification code directly;
   *System 2* first writes a short structured rationale.
4. **Triage** — deterministic rules pick the primary finding by severity
   (a single maladaptive span forces a correction; unclear is never safe).
5. **Constraint synthesis** — a "validate + correct" instruction is appended
   to the generator's prompt: acknowledge the adaptive parts of the proposal,
   redirect the maladaptive one toward the ontology's corrective alternatives.

The ontology (`safety/graph_definitions.py`) has 13 nodes and 15 corrective
edges: five maladaptive patterns (avoidance, interruption, violent revenge,
suppression, unchanged trauma replay), the five mastery categories of the
Multidimensional Mastery Scale (behavioral, social, environmental, emotional,
mythical), two neutral categories (narrative setting, affect expression), and
`UNCLASSIFIED` as the fail-closed fallback. The evaluated configuration uses
Mistral NeMo Instruct 2407 as segmenter and Mistral Small 3.2 24B Instruct
2506 as classifier, both through Scaleway at temperature 0; the
classification-consistency variant adds five sampled calls at temperature 0.7.

On the development benchmark (420 scored cases) the default System 1 pipeline
reaches 97.9 % safety accuracy (411/420) and 99.4 % unsafe recall. The thesis
reports the full evaluation, including the held-out test set, the ablations,
the uncertainty analysis and a blinded expert evaluation.

## Repository layout

```
AI/                       baseline IRT chatbot (FastAPI server, terminal client,
                          optional Streamlit UI); the safety layer is invoked in
                          the rewriting stage of AI/irt_app.py
safety/                   the safety layer: ontology, classifiers, critic pipeline
experiments/              benchmark runner, dataset generators, diagnostics, and
  evaluation_statistics/  the scripts that compute every Chapter 4 statistic
data/
  benchmarks/             datasets and result files (see data/benchmarks/README.md)
  expert_spotcheck/       blinded spot-check instrument and answer key
  system_eval/            system-level evaluation: scenarios, chatbot responses in
                          both conditions, blinded rating instrument and key
  integration_eval/       internal constraint/redirect pilot
  evaluation_statistics/  audit bundle: canonical statistics, provenance, audits
  scenarios/              small scenario suites for the interactive tool
  worked_example/         recorded outputs of the two worked examples (appendix)
docs/                     architecture deep dive, ablation specification, clinical
                          rationale of the ontology, ontology review document (DE)
tests/                    unit, regression and statistics tests
```

## Setup

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
uv sync                     # add --extra ui for the Streamlit interface
cp .env.example .env        # then fill in the keys you need
```

| Key | Needed for |
| --- | --- |
| `SCALEWAY_API_KEY` | the safety layer (segmenter + classifier) and the chatbot's routing and response models |
| `MISTRAL_API_KEY` | the chatbot's fallback models (Mistral API) |
| `OPENROUTER_API_KEY` | dataset generation with GPT-5.4 only |
| `LANGFUSE_*` | optional tracing (`docs/architecture/langfuse_tracing.md`) |

Model identifiers and providers are listed in `AI/agent.py` (`MODELS`).

## Running the chatbot

```bash
uv run uvicorn AI.api:app --reload             # FastAPI server on :8000
uv run python -m AI.chat_client --stream       # terminal client
uv run --extra ui streamlit run AI/streamlit_app.py
```

Endpoints: `POST /chat`, `POST /chat/stream`, `GET /conversation/{session_id}`,
`POST /chat/set_history`, `GET /health`. The safety critic is initialised
fail-open: if it cannot start (for example without a key), the chatbot runs
without it and logs a warning. Response payloads carry the safety result under
`safety`.

## Using the safety layer directly

```python
import asyncio
from safety.critic import SafetyCritic

async def main():
    critic = SafetyCritic()
    result = await critic.evaluate_intervention(
        "I call my brother for help, then I burn the whole city down.",
        nightmare_context="I am chased through a burning city and cannot get out.",
    )
    print(result.is_safe, result.node_id, result.severity)   # False VIOLENT_REVENGE CRITICAL
    print(result.constraint)   # the instruction appended to the generator prompt
    for segment in result.segments:
        print(segment.text, "->", segment.node_id, round(segment.confidence, 3))

asyncio.run(main())
```

`SafetyResult.to_dict()` gives the serialisable form; `data/worked_example/`
holds the recorded output of exactly this example. For a walk through a whole
scenario suite with per-segment output:

```bash
uv run python tests/test_critic_interactive.py --scenario data/scenarios/chase_scenario_tests.json -v
```

## Reproducing the thesis statistics

No API key is needed; everything is recomputed from the row-level records in
`data/`.

```bash
uv run python experiments/evaluation_statistics/recompute_chapter4.py
uv run python experiments/evaluation_statistics/verify_chapter4_statistics.py
for script in experiments/evaluation_statistics/chapter_4/*.py; do uv run python "$script"; done
```

The first command rewrites `data/evaluation_statistics/chapter4/` (canonical
`chapter4_statistics.json`, a Markdown headline report, two small LaTeX macro
and table files used by the thesis, and `provenance.json` with the SHA-256 of
every input). The second is a deliberately simple standalone implementation
that recomputes the values with visible formulas and checks them against the
canonical JSON. The section scripts print each section's numbers in thesis
order with their numerators and denominators.

**Withheld data.** The row-level annotations of the two clinical experts are
personal data and are not part of this release; they will be added if the
raters consent. The recompute therefore reproduces every benchmark-derived
value and carries the four expert-derived blocks over from the shipped
canonical JSON, recording this under `withheld_inputs` in `provenance.json`;
the verifier skips the corresponding checks and says so in its report; the
scripts for §4.3 and §4.8 print a notice instead of results. The evaluation
instruments the experts used (blinded cases, rating apps, answer keys) are
included.

## Re-running the experiments

These call hosted models and cost money; the recorded outputs are already in
`data/benchmarks/`. `data/benchmarks/README.md` maps every result file to its
configuration, run date and thesis section.

```bash
uv run python experiments/run_benchmark.py --help
uv run python experiments/run_benchmark.py                                   # dev set, System 1
uv run python experiments/run_benchmark.py --classifier s2 --consistency 5   # System 2 + consistency votes
uv run python experiments/run_benchmark.py --no-segment                      # ablation: no segmentation
uv run python experiments/run_benchmark.py --input data/benchmarks/test_dataset.json --output data/benchmarks/test_s1_baseline.csv
```

Dataset generation (`experiments/generate_dataset.py`,
`experiments/generate_multi_maladaptive_slice.py`) used GPT-5.4 through
OpenRouter. Hosted models change over time, so re-runs reproduce the
documented configuration, not necessarily the recorded numbers; the thesis
discusses the observed run-to-run variation.

## Tests

```bash
uv run pytest tests
```

Tests that call hosted models are skipped unless `SCALEWAY_API_KEY` is set;
tests that need the withheld expert tables are skipped when the tables are
absent. Everything else runs offline.

## Provenance

This repository is a snapshot of commit `5783dd4` (2026-09-28)
of the private development repository, exported on 2026-09-28; the
development history is not included. The code in `safety/` and `AI/` is the
state used for the reported evaluations. The thesis appendix reproduces the
classifier and segmenter prompts exported from these files. The README files
inside the `data/` and `experiments/` folders document the full research
repository; files they list as archived, superseded, expert annotations, or
under `thesis/` are not part of this release.

## License

- Source code: [Apache License 2.0](LICENSE).
- Datasets, result files, evaluation materials and documentation under
  `data/` and `docs/`: [Creative Commons Attribution 4.0
  International](LICENSE-CC-BY-4.0).

See [NOTICE](NOTICE). The name *ReDreamAI* is not licensed.

## Citation

Please cite the thesis (see [CITATION.cff](CITATION.cff)):

> Zacharias, R. Y. (2026). *A Neuro-Symbolic Safety Layer for LLM-Assisted
> Imagery Rehearsal Therapy* [Master's thesis, Osnabrück University, Institute
> of Cognitive Science].

Code and data: <https://github.com/RZacharias44/redream-safety-layer>.

## Clinical background

- Germain, A., Krakow, B., Faucher, B., Zadra, A., Nielsen, T., Hollifield, M.,
  Warner, T. D., & Koss, M. (2004). Increased mastery elements associated with
  imagery rehearsal treatment for nightmares in sexual assault survivors with
  PTSD. *Dreaming, 14*(4), 195–206. — the Multidimensional Mastery Scale.
- Krakow, B., & Zadra, A. (2006). Clinical management of chronic nightmares:
  Imagery rehearsal therapy. *Behavioral Sleep Medicine, 4*(1), 45–70. — the
  IRT protocol.

`docs/clinical/Clinical_Ontology_Scientific_Rationale.md` gives the rationale
of every ontology node with its sources.
