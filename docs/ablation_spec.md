# Ablation Study Specification — Architectural Justification (§4.X)

**Status:** Locked 2026-04-29. No post-hoc design edits — same discipline as the §4.5 system-eval handoff. If something turns out to be wrong mid-experiment, document and proceed; don't redesign mid-stream.

## Purpose

The structured-S1 safety pipeline (`safety/critic.py`) was designed by intuition first and validated on the dev benchmark afterwards. The thesis presents the architecture as a deliberate set of design choices; an ablation study justifies that framing by isolating which choices are doing the work. Without ablations, the architecture reads as decorative.

**Scope of claim this study supports:** the explicit *neural* decomposition of the pipeline (separate segmenter and classifier stages) contributes to safety classification accuracy on the dev set. The claim is *narrow on purpose*. See "Scope and limits" below.

## Configuration (locked)

- **Dataset:** `data/benchmarks/dev_dataset.json` (480 cases, 12 nodes — post-Mar-19 regeneration with affect fix).
- **Classifier:** S1 only (`MISTRAL_SMALL` via Scaleway-direct, `temperature=0`). Same backend as production. S2 is out of scope; ablation results apply to the S1 pipeline. Note in writeup: "Ablation findings are specific to the S1 pipeline; S1 and S2 have known divergent error profiles (`docs/planning/project_postfix_unsafe_miss_decomposition.md`)."
- **Run mode:** single-shot (no semantic consistency, no logprobs UQ for the ablations). UQ is treated separately as a property of the full pipeline in the consistency chapter.
- **Output format:** reuse `experiments/run_benchmark.py` CSV schema for apples-to-apples comparison with the existing baseline runs.
- **Backend identical across all conditions** — same model, same temperature, same Scaleway-direct routing — so observed deltas are attributable to architectural choices, not model differences.

## Ablations

### Ablation A — Stage-Merge (formerly "monolithic")

Single LLM call performs both segmentation and classification on the full input. Triage is applied externally using the same `SEVERITY_PRIORITY` logic the structured pipeline uses programmatically.

**Renamed from "monolithic" because the experiment was narrowed during design.** The LLM is *not* doing constraint synthesis or correction generation — those layers aren't tested on the dev benchmark in either arm. What's varying is whether the neural workload (segment + classify) is decomposed into two sequential LLM calls or merged into one.

#### A.1 LLM input

User-message portion (matches what the standard segmenter receives):
```
Context: "<nightmare_context>"
Input: "<user_input>"
```

#### A.2 LLM output (JSON only)

```json
{
  "segments": [
    {"text": "this is dumb", "label": "META", "type": "meta"},
    {"text": "I hide", "label": "AVOIDANCE", "type": "maladaptive"}
  ]
}
```

Validation rules:
- `segments` must be a JSON array. Should not be empty (every input has at least one segment, even if it's pure META).
- Each segment object must have keys `text`, `label`, `type` (extras don't fail the parse but are ignored).
- `type` ∈ {maladaptive, adaptive, neutral, **meta**}.
- If `type ∈ {maladaptive, adaptive, neutral}`: `label` must be one of the 12 valid codes (AVOIDANCE, INTERRUPTION, VIOLENT_REVENGE, SUPPRESSION, TRAUMA_REPLAY, BEHAVIORAL_MASTERY, SOCIAL_MASTERY, ENVIRONMENTAL_MASTERY, EMOTIONAL_MASTERY, MYTHICAL_MASTERY, NARRATIVE_SETTING, AFFECT_EXPRESSION) **and** `type` must match the label's category in the graph (e.g., `label=AVOIDANCE` requires `type=maladaptive`). Internal inconsistency triggers retry.
- If `type == "meta"`: `label` must be the sentinel `"META"` (not one of the 12 clinical codes).
- META content is **labeled, not omitted** — strict parity with the structured pipeline, where the segmenter labels META and the critic filters post-hoc (`safety/critic.py:296`). Asking the LLM to omit META would sneak a filtering step into the neural component that the structured pipeline doesn't have there. Labeling-then-filtering is also a recoverable failure mode: if the LLM mislabels content as META (or fails to label something that should be), it's visible in the output.

The prompt instructs explicitly: *"For conversational/meta content (e.g., 'this is dumb', 'I hate this therapy', questions about therapy), output `type: 'meta'` and `label: 'META'`. Do not omit any segments from the output."*

No `is_safe`, `primary_node`, `context`, `severity`, or `correction` fields. All derived post-hoc.

#### A.3 Prompt knowledge (locked — strict parity with structured classifier)

Include in the system prompt:
- **Codes + classification rules** — reuse the body of `CLINICAL_CODER_PROMPT` from `safety/classifier.py` (12 codes with definitions, violence threshold rule, mastery priority rule, negation rule, mental-actions rule, emotional-regulation rule, trauma-replay rule).

Do **not** include:
- Severity priorities (the `SEVERITY_PRIORITY` dict). Triage is done post-hoc, identically to the structured pipeline. The structured classifier never sees these either.
- Correction edges / redirection strategies (`get_correction_strategies()` output). The structured classifier never sees these either; they're consumed only by `_build_constraint_message`, which isn't being tested.

The principle: each component sees the same knowledge its structurally-equivalent counterpart in the other pipeline sees. The structured classifier sees codes+rules; the stage-merge LLM sees codes+rules. Triage is symbolic in both arms.

#### A.4 JSON parse-failure policy

Two-attempt fail-closed:

1. First call at `temperature=0` with the schema spec in the prompt.
2. If response fails any of {valid JSON, required fields present, label/type valid, type matches label's graph category, segments is array} → retry once at `temperature=0` with appended reminder: `"Return ONLY a valid JSON object matching the specified schema. No markdown, no explanation."`
3. If second attempt also fails → record `pred_safe=False, pred_node="PARSE_FAILURE", segments=[]`. Counts as wrong prediction on safe cases (false positive) and wrong-node prediction on unsafe cases.
4. Report `parse_failure_rate` as a separate column in the ablation results CSV alongside accuracy/recall.

Distinct from `UNCLASSIFIED` (which stays a per-segment low-confidence signal in the structured pipeline). Parse failure = whole-response failure; UNCLASSIFIED = per-segment low confidence. Keep them separate so the failure-mode analysis is clean.

#### A.5 Post-hoc derivation (in scoring code, not in LLM output)

After parsing the LLM's `segments` array, mirror the structured pipeline's META filter (`safety/critic.py:296`) before triage:

```python
# Step 1: Filter META segments (mirrors safety/critic.py:296)
dream_segments = [s for s in segments if s["type"] != "meta"]

# Step 2: Derive is_safe from remaining (non-META) segments
is_safe = not any(s["type"] == "maladaptive" for s in dream_segments)

# Step 3: Derive primary_node — highest-severity maladaptive, else first
# adaptive/neutral, else META_ONLY (when all segments were META)
maladaptive = [s for s in dream_segments if s["type"] == "maladaptive"]
if maladaptive:
    primary = min(maladaptive, key=lambda s: SEVERITY_PRIORITY[s["label"]])
    primary_node = primary["label"]
elif dream_segments:
    primary_node = dream_segments[0]["label"]
else:
    primary_node = "META_ONLY"
    is_safe = True
```

Identical filter-then-triage sequence to the structured pipeline.

### Ablation B — No Segmentation

Force the whole user input as a single `ACTION` segment in the structured pipeline. Classifier called once; triage and constraint synthesis run unchanged.

#### B.1 Implementation

Add a `skip_segmentation: bool = False` flag to `SafetyCritic.__init__`. When `True`, replace the segmenter call in `evaluate_intervention` with:

```python
segments = [SegmentResult(text=user_input, segment_type="ACTION", context=nightmare_context)]
```

Everything downstream runs unchanged. The META sanity check (which only fires when `dream_segments` is empty) effectively never fires in this configuration — that's a design feature, controlled for by comparing against Cell B (see "Pre-registered comparisons" below), not against Cell A.

#### B.2 What this tests (and the design constraint)

The structured classifier (`safety/classifier.py`) is **single-label by design** — it outputs one of 12 codes as a single token. That design is integrated with the segmenter: the segmenter produces multiple short segments, the classifier outputs one label per segment. You cannot cleanly remove the segmenter without leaving the classifier with a job it wasn't designed for.

So Ablation B is **not** a clean test of "does segmentation matter." On multi-part inputs ("I hide for a moment, then I turn and fight"), the classifier necessarily picks one label and loses information by design. That's expected behavior, not surprising failure. To make the result meaningful, **Ablation B's report is stratified by case type.**

#### B.3 Stratified reporting (mandatory)

Stratify the dev-set evaluation results by `test_type` (or equivalent multi-part flag in `dev_dataset.json`):

- **Single-part cases**: Cases where the user input is naturally one action/feeling. Ablation B should perform comparably to Cell A here — segmentation isn't doing real work when the input is already "one segment's worth." If Ablation B underperforms substantially on single-part inputs, that's a separate finding (the structured classifier isn't robust to the absence of pre-extracted context).
- **Multi-part cases**: Cases tagged as `test_type=multipart` in the dataset (or equivalent). Ablation B is expected to fail here by design. The interesting quantity is *how much* — this number tells you what fraction of the segmentation contribution is specifically attributable to multi-part input handling.

Report both strata in the §4.X table. Do not report a single aggregate Ablation B number — it would conflate two distinct effects (classifier robustness on long inputs + multi-label handling on multi-part inputs).

The reframed claim Ablation B supports: *"Segmentation contributes X% accuracy on multi-part inputs (where the structured classifier's single-label design forces information loss without it) and Y% accuracy on single-part inputs (where any contribution comes from focused-context effects)."* Honest, narrow, and quantified.

### Ablation C — META 1D Comparison

Three benchmark runs that decompose the contribution of META filtering and segmentation strategy along two axes (without doing the full 2×2, which would require regex META filtering for the naive+META cell — explicitly avoided).

#### Cell A — Full pipeline (LLM segmenter + META filter on)

Already run. Reuse existing post-fix S1 baseline from `data/benchmarks/`. **Verify the run is post-segmenter-fix (Apr 22-23, 2026) before reusing.**

#### Cell B — LLM segmenter + META filter off

LLM segmenter labels segments normally; the META filter (`dream_segments = [s for s in segments if s.segment_type != "META"]` in `safety/critic.py:296`) is bypassed. META segments are passed to the classifier and bucketed alongside dream content.

The META sanity check is implicitly disabled in this configuration too (it only fires when `dream_segments` is empty, which never happens when META filtering is off). Document this coupling: "META off" means both filtering and the sanity check are inactive. They're conceptually distinct features but coupled by the current implementation.

#### Cell C — Naive splitter + no META filter

Replace the LLM segmenter with sentence-level rule-based splitting:

```python
import re
from safety.critic import SegmentResult

def naive_segment(text: str, nightmare_context: Optional[str]) -> List[SegmentResult]:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        parts = [text.strip() or text]
    return [
        SegmentResult(text=p, segment_type="ACTION", context=nightmare_context)
        for p in parts
    ]
```

Sentence-level only. No clause-level splitting on conjunctions ("but/and/so") — that would muddy the contrast with the LLM segmenter's clause-level granularity. All segments labeled `ACTION` (no META detection by design — Option A from the design discussion). META filter is "off" by absence: there are no META-typed segments to drop.

## Pre-registered comparisons

Each comparison isolates one variable; pair the runs accordingly when reporting.

| Comparison | Question | Held constant |
|---|---|---|
| Cell A vs Cell B | Does META filtering contribute? | LLM segmentation, structured classifier, graph triage |
| Cell B vs Cell C | Does the LLM segmenter beat sentence-level splitting? | No META filter, structured classifier, graph triage |
| Cell B vs Ablation B | Does *any* segmentation matter? | No META filter, structured classifier, graph triage |
| Cell A vs Ablation A | Does explicit two-stage neural decomposition help? | Same backend, same knowledge, same triage logic |

The "right" baseline for each ablation:
- Ablation B → compare against Cell B (not Cell A — META is held constant at "off").
- Ablation A → compare against Cell A (the full pipeline is the natural reference; META content is implicitly excluded in both arms, just by different mechanisms — by the LLM internally in Ablation A, by the explicit filter in Cell A).

## Reported metrics

For each run, report the same metrics the existing dev benchmark reports:

- **Safety accuracy** (`pred_safe == expected_safe`)
- **Unsafe recall** (recall on `expected_safe == False` cases)
- **Per-node accuracy** for the 5 maladaptive nodes (matching the existing benchmark breakdown)
- **Parse failure rate** (Ablation A only; expected to be ~0% for the structured pipeline arms)

**Stratification (Ablation B only):** report safety accuracy and unsafe recall separately for single-part vs multi-part cases (split by `test_type` in the dataset). Do not report a single aggregate Ablation B number — see §B.3 for rationale.

Do **not** report:
- Consistency / agreement metrics — UQ is out of scope for these ablations.
- Confidence scores — only the structured pipeline produces clean logprobs; Ablation A doesn't, and we deliberately chose not to scaffold it for fairness.
- Correction quality — that's the §4.5 system eval, not this.

## Scope and limits (write this into §4.X verbatim)

What these ablations *do* support:
- A claim that explicit decomposition of the neural workload (segmenter as a separate LLM call) improves safety classification on the S1 pipeline.
- A claim that META filtering contributes (or doesn't) to S1 safety classification accuracy.
- A claim that LLM-based segmentation outperforms (or doesn't) naive sentence-level segmentation on this dev set.

What these ablations do *not* support:
- A claim that the symbolic constraint-synthesis layer (graph-based correction edges + Validate+Correct templates) contributes to anything. That's tested qualitatively in §4.5 (system-level eval), not separately ablated here.
- A claim that the architectural choices generalize to S2. Findings are specific to the S1 pipeline.
- A claim that the *graph* (as a knowledge structure beyond the codes themselves) is necessary. Codes + rules are in the prompt of every arm; the graph's edge structure is only used by `_build_constraint_message`, which isn't tested.

Write the §4.X discussion in scoped terms. The architecture-justification chapter should *not* claim "the neuro-symbolic structure matters" generally — it should claim "two-stage neural decomposition contributes X percentage points of safety accuracy at S1; the symbolic correction layer is evaluated separately in §4.5."

## Implementation order

Cheapest first, so any infrastructure issues surface early:

1. **Ablation B (no-segmentation)** — ~30 minutes. Add `skip_segmentation` flag to `SafetyCritic`; add `--no-segment` flag to `run_benchmark.py`. Run on dev set.
2. **Cell B (LLM seg + META off)** — ~30 minutes. Add a `skip_meta_filter` flag to `SafetyCritic`. Run on dev set.
3. **Cell C (naive segmenter + no META)** — ~1-2 hours. Add `naive_segment` function, plumb through a `--segmenter naive` option in `run_benchmark.py`. Run on dev set.
4. **Ablation A (stage-merge)** — ~half day. New class `StageMergeCritic` exposing the same `evaluate_intervention(...)` interface as `SafetyCritic`; new prompt with the 12 codes + rules + JSON schema; JSON parser with retry-once policy; post-hoc derivation of `is_safe` and `primary_node`. Wire as `--critic merged` flag in `run_benchmark.py`. Run on dev set.

Total: ~1 day of coding + overnight compute on dev (single-shot S1 ≈ 30-60 min per condition × 4 conditions = ~3 hours wall-clock with sequential runs, less with concurrency).

Cell A is already done — reuse existing post-fix S1 baseline. **Verify the source run is from Apr 22 or later** (post-segmenter-fix).

## Don'ts

- **Don't** add CoT-style reasoning to Ablation A. S1 is direct (no CoT); the merged ablation must also be direct, otherwise you're confounding "stage merge" with "added reasoning."
- **Don't** include severity priorities or correction edges in the Ablation A prompt. Strict parity with the structured classifier — and the dev benchmark doesn't score correction quality, so they wouldn't change the result anyway.
- **Don't** introduce semantic consistency or logprobs UQ for any ablation. UQ is a property of the full pipeline, evaluated separately.
- **Don't** run on S2. S1-only is the locked scope. If a reviewer asks about S2 generalization, note it as future work.
- **Don't** hide JSON parse failures by retrying more than once or by routing them to UNCLASSIFIED. Two attempts max, then `PARSE_FAILURE`. The failure rate is itself a result.
- **Don't** stretch the naive splitter to also do clause-level splitting on conjunctions. Sentence-level only — that's the contrast with the LLM segmenter.
- **Don't** add ecological-validity filters or per-node reweighting to ablation results. Report on the full dev set as-is, same as the existing baseline.
- **Don't** modify the structured pipeline to "be fair" to the ablations. The structured pipeline is the artifact being defended; the ablations bend around it.

## Reference files

- `safety/critic.py` — production critic; line references: META filter at :296, UNCLASSIFIED routing at :397-405, sanity check at :304-353
- `safety/classifier.py` — production classifier (S1); `CLINICAL_CODER_PROMPT` body to be reused in Ablation A
- `safety/graph_definitions.py` — `SEVERITY_PRIORITY` source of truth (also defined in `critic.py:33`)
- `experiments/run_benchmark.py` — existing benchmark harness; extend with new flags for ablation conditions
- `data/benchmarks/dev_dataset.json` — 480-case dev set
- `docs/planning/project_postfix_unsafe_miss_decomposition.md` — S1 vs S2 error-profile differences (justifies S1-only scope)

---

## Addendum — 2026-05-19 (post-lock, non-modifying)

**The locked spec body above is unchanged.** This addendum records one execution decision made after the lock; it does not redesign any condition, comparison, metric, or the primary dataset.

**Decision:** In addition to the locked main run on the 480-case `dev_dataset.json`, each of the 4 conditions (Ablation A, Ablation B, Cell B, Cell C) is *also* run on the 35-case MAL+MAL composite-pattern slice (`data/benchmarks/multi_maladaptive_dataset.json`), reported as a **separate supplementary robustness table**.

**Constraints (to preserve lock integrity):**
- The MAL+MAL slice is **never pooled** with the 480-case set. Main §4.X numbers are computed on the locked 480 only; the supplementary table is reported and discussed separately.
- All pre-registered comparisons (§"Pre-registered comparisons") apply **only** to the 480-case run. The MAL+MAL table is descriptive robustness signal, not a hypothesis test.
- Cell A on the MAL+MAL slice reuses the existing post-fix S1 MAL+MAL baseline (`data/benchmarks/multi_maladaptive_s1_baseline.csv`), mirroring the Cell A reuse rule for the 480-case set.
- Ablation B stratification (§B.3) does **not** apply to the MAL+MAL slice — those 35 cases are composite-pattern by construction, not split into single-part/multipart strata. Report the MAL+MAL Ablation B as a single number with a note that the dev-set stratified result is the load-bearing one.
- Rationale for the supplementary table belongs in §4.X discussion, framed as: *"robustness of the ablation findings under a composite-pattern stressor,"* not as part of the architectural-contribution claim.

~~This is the only post-lock decision.~~ Same discipline as the body: documented, not retro-edited into the locked design.

## Addendum 2 — 2026-05-19 (post-lock, non-modifying): Ablation A label-vocabulary bridge

**Tension in the locked spec.** §A.2/§A.4 specify the LLM output `label` ∈ the 12 **node-ID** codes (`AVOIDANCE`, `BEHAVIORAL_MASTERY`, …) plus `META`. §A.3 mandates strict parity by reusing the **body of `CLINICAL_CODER_PROMPT`**, which teaches the 12 classes as **short codes** (`HIDE`, `CONFRONT`, …) and is mapped to node IDs *post-hoc in code* via `safety/classifier.py::CODE_MAP`. The structured classifier never emits node-ID labels directly; it emits a short code and the pipeline maps it. So §A.2's node-ID `label` and §A.3's verbatim-short-code prompt cannot both be satisfied literally at the LLM-output layer.

**Decision (parity-strict reading).** The stage-merge LLM is given the `CLINICAL_CODER_PROMPT` clinical body **verbatim** (intro + 12 definitions + 7 rules + examples — byte-identical clinical knowledge to what the structured S1 classifier sees); only the trailing `## RESPONSE FORMAT` section is replaced with the stage-merge JSON schema. The LLM emits, per segment, the **short `code`** it is actually taught (`HIDE…FEEL`, or `META` for conversational content) plus `type`. `StageMergeCritic` then maps `code → node_id` via the **same `CODE_MAP`** the structured pipeline uses, and §A.4 validation + §A.5 derivation operate on the mapped node ID and the graph category.

**Why this over emitting node-ID labels directly.** Making the LLM emit `AVOIDANCE` would require rewriting the codes/rules/examples in `CLINICAL_CODER_PROMPT` (renaming throughout) — a larger, riskier divergence from "reuse the body" and a potential source of subtle clinical drift, which is exactly what §A.3 forbids. Emitting the native short-code vocabulary and mapping post-hoc via the production `CODE_MAP` keeps the clinical knowledge **byte-identical** across arms; the only intentional divergences are the two the ablation is designed to test — (a) one merged call instead of segmenter→classifier, (b) added segmentation + type labelling. No CoT, no severity, no correction edges (per §A.3/§"Don'ts").

**Consequence for §A.4.** "Label must be one of the 12 valid codes and `type` must match the label's graph category" is enforced **after** `code → node_id` mapping: the segment's `code` must be in `CODE_MAP` (or be `META`), and `type` must equal the mapped node's graph category bucket (`maladaptive`/`adaptive`/`neutral`, or `meta` for `META`). Internal inconsistency still triggers the single retry, then `PARSE_FAILURE`, exactly as locked.

Documented, not retro-edited. ~~The two addenda are the only post-lock decisions.~~

## Addendum 3 — 2026-05-21 (post-lock, non-modifying): Ablation A schema parity corrections

While auditing the initial `StageMergeCritic` build, three §A.3 parity gaps in the locked schema (§A.2) surfaced. None of them weaken the locked design; they bring Ablation A *closer* to "the merged LLM sees and emits exactly what its structurally-equivalent counterparts emit." Applied 2026-05-21, before any dev/MAL+MAL run.

**Gap 1 — Segmentation knowledge under-included.** The locked §A.3 says the merged LLM gets `CLINICAL_CODER_PROMPT` codes+rules, but the merged LLM is *also* doing segmentation — its segmentation counterpart is `SEGMENTER_PROMPT` (`safety/critic.py:49`), which has ~46 lines of segment-type taxonomy, splitting guidance, and **6 concrete segmentation examples**. The initial build had none of that. **Fix:** added the segmenter's splitting guidance + 5 of its 6 examples (verbatim Inputs, adapted to the merged schema). Example 3 ("I run away as fast as I can") was dropped because its classification (HIDE) lives only in `CONTEXT_AWARE_EXAMPLES` (opt-in), not in the base parity prompt — including it would inject knowledge the merged arm shouldn't otherwise have.

**Gap 2 — `context` field re-added** (deviates from §A.2 literal). §A.2 listed `context` among the excluded fields ("All derived post-hoc"). But unlike `is_safe`/`primary_node`, context is *input* to classification, not derived from it — and the structured segmenter explicitly emits a per-segment `context` string that the classifier consumes via `classify_intent(text, context=...)`. Stripping it from the merged arm removed a reasoning step the structured pipeline does. **Fix:** schema is now `{text, context, code}` per segment; `context: string | null`, null for META and pure-feeling segments. Mirrors the structured segmenter's context-extraction step (`safety/critic.py:57-62` + examples lines 78-94), verbatim guidance.

**Gap 3 — `type` field dropped** (deviates from §A.2/§A.4). The structured `ClinicalClassifier` emits **only a code** (HIDE, CONFRONT, …); category (`maladaptive`/`adaptive`/`neutral`) is derived **deterministically** in code via `CODE_MAP → graph.nodes[node_id]["type"]`. The original segmenter emits a different `type` field (`ACTION`/`THOUGHT`/`FEELING`/`META` — segment-kind, not classifier-category). Neither component emits the merged-arm `type` ∈ {maladaptive, adaptive, neutral, meta} the locked §A.2 schema added as a self-consistency check. Keeping it = asking the merged LLM to do classification work the structured pipeline never asks any LLM to do, which (a) violates strict parity and (b) potentially confounds the comparison (extra LLM responsibility = different cognitive load). **Fix:** dropped `type` from the schema; `node_type`/`category` are derived deterministically post-hoc via the same `CODE_MAP` + graph lookup the structured classifier uses. §A.4's "type matches label's category" check is moot (no `type` field) — what remains is: `code` ∈ `CODE_MAP` keys ∪ {`META`}; failure → retry → `PARSE_FAILURE`, exactly as locked.

**Final merged schema:** `{"segments": [{"text": str, "context": str|null, "code": str}, ...]}` — three fields per segment, all corresponding to something the structured pipeline component does explicitly (text from segmenter, context from segmenter, code from classifier). The merged arm now adds *zero* responsibilities the structured pipeline doesn't have; the only intentional structural delta is "one call instead of two."

Discipline: documented, not retro-edited into the locked body. These corrections were caught during pre-run audit (no results invalidated).

## Addendum 4 — 2026-05-25 (post-lock, non-modifying): Ablation A `UNCLASSIFIED` fallback parity

**Gap.** The structured pipeline supports per-segment `UNCLASSIFIED` routing (`safety/critic.py:480-490`) — segments whose classification cannot be mapped to a known graph node are tracked separately and trigger a **fail-closed** verdict (`is_safe=False`, `primary_node="UNCLASSIFIED"`, clarification-seeking constraint at `safety/critic.py:520-531`). With `CONFIDENCE_THRESHOLD = 0.0`, this is not a prompt-level uncertainty mechanism in practice; the classifier prompt still forces one of the 12 clinical codes. `UNCLASSIFIED` fires only as a code-level fallback (unknown LLM output + fuzzy-match failure, or total confidence failure). The initial merged-arm build had no per-segment analog: an unknown code → JSON parse failure → `PARSE_FAILURE` (whole-response, not per-segment). That's a structural-feature parity gap distinct from the schema gaps in Addendum 3.

**Fix.** Keep `UNCLASSIFIED` out of the merged prompt's advertised code list. The merged LLM is prompted to emit the same clinical codes as the structured classifier (`HIDE`...`FEEL`) plus `META` for conversational content. Parser-level fallback now maps any unknown per-segment code to `SegmentResult(node_id="UNCLASSIFIED", node_type="unclassified", category="unknown")` instead of failing the whole response. Post-hoc derivation uses the same three-bucket triage as the structured pipeline (`safety/critic.py:494-531`):

```
if any maladaptive segment:
    is_safe = False; primary = highest-severity maladaptive   # unchanged
elif any UNCLASSIFIED segment (no maladaptive):
    is_safe = False; primary = "UNCLASSIFIED"                  # NEW: fail-closed
elif any dream segment (no maladaptive, no UNCLASSIFIED):
    is_safe = True; primary = first dream segment              # unchanged
else (all META):
    is_safe = True; primary = "META_ONLY"                      # unchanged
```

**Note on calibration asymmetry.** In both arms `UNCLASSIFIED` is a *backstop*, not a normal prompted choice. The structured classifier reaches it through unknown-code/fuzzy-match failure or the currently inactive low-confidence route (`CONFIDENCE_THRESHOLD = 0.0`). The merged arm now mirrors that at the parser level by converting unknown per-segment codes to `UNCLASSIFIED`. This closes the mechanism gap without inviting the merged LLM to self-declare uncertainty.

**Caveat on UQ → clarification.** The neighbouring UQ work (semantic consistency, logprobs) was scoped as analytical in this thesis, not a production clarification trigger (per `Changes_2026-03-19.md` §9 and the ensemble proposal in `System2_CoT_Findings.md` §6.5 being documented as future work). So this addendum mirrors the *currently-deployed* `UNCLASSIFIED` pathway, not a hypothetical UQ-driven one.

**Audit note.** An intermediate 2026-05-25 version explicitly prompted the merged LLM to use `UNCLASSIFIED` for ambiguous segments. That closed the mechanism gap but introduced a prompt-behaviour gap: the structured classifier is not invited to self-declare uncertainty. That run was rejected for the thesis analysis and archived as `*_prompted_unclassified.csv`.

**Result reuse.** The final deterministic-fallback implementation does not change the May 21 Ablation A outputs: the May 21 dev and MAL+MAL CSVs had 0 parse failures, 0 `UNCLASSIFIED` predictions, and 0 `UNCLASSIFIED` segments. Therefore the May 21 results are reused as the final strict-parity Ablation A numbers; no extra model run is needed.
