"""
Semantic Consistency Uncertainty Quantification

Runs N stochastic inference chains at elevated temperature and measures
classification agreement as a confidence proxy.

Works with ANY classifier that implements the classify_intent() interface
(both System 1 ClinicalClassifier and System 2 ClinicalClassifierCoT).

Research question: Does semantic consistency better predict classifier errors
than token log-probabilities, and thus provide a more reliable trigger for
requesting user clarification?
"""

import asyncio
import logging
import time
from typing import Optional, Tuple, List, Protocol
from dataclasses import dataclass, field
from collections import Counter

from .classifier import ClassificationResult, CODE_MAP

logger = logging.getLogger(__name__)


class ClassifierProtocol(Protocol):
    """Protocol that both System 1 and System 2 classifiers satisfy."""

    async def classify_intent(
        self,
        user_text: str,
        context: Optional[str] = None,
        return_logprobs: bool = False
    ) -> Tuple[str, Optional[ClassificationResult]]:
        ...


@dataclass
class ConsistencyResult:
    """Result of semantic consistency evaluation."""
    # The majority-vote classification
    node_id: str
    code: str
    # Consistency metrics
    agreement_ratio: float  # Fraction of runs agreeing with majority (0-1)
    n_runs: int
    vote_distribution: dict  # {code: count}
    # Individual run results (for analysis)
    individual_results: List[Tuple[str, str]]  # [(node_id, code), ...]
    # Whether this is considered "confident" (agreement >= threshold)
    is_confident: bool
    # The ClassificationResult from the deterministic (low-temp) run
    deterministic_result: Optional[ClassificationResult] = None
    # Computational cost metrics (totals across all runs: 1 deterministic + N stochastic)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_ms: float = 0.0


async def evaluate_semantic_consistency(
    classifier: ClassifierProtocol,
    user_text: str,
    context: Optional[str] = None,
    n_runs: int = 5,
    stochastic_temperature: float = 0.7,
    confidence_threshold: float = 0.8,
) -> ConsistencyResult:
    """
    Evaluate classification confidence via semantic consistency.

    Runs the classifier N times at elevated temperature and measures
    agreement across classifications. High agreement = confident;
    low agreement = uncertain.

    Args:
        classifier: Any classifier implementing classify_intent()
        user_text: The text to classify
        context: Optional nightmare context
        n_runs: Number of stochastic runs (default 5)
        stochastic_temperature: Temperature for stochastic runs (default 0.7)
        confidence_threshold: Agreement ratio threshold for "confident" (default 0.8)

    Returns:
        ConsistencyResult with majority vote and agreement metrics
    """
    # Save original temperature
    original_temp = classifier.agent.temperature

    # Track cumulative cost across all runs
    total_input_tokens = 0
    total_output_tokens = 0
    t_start = time.perf_counter()

    # Step 1: Run deterministic classification (original temperature) with logprobs
    classifier.agent.temperature = original_temp
    det_node_id, det_result = await classifier.classify_intent(
        user_text, context=context, return_logprobs=True
    )
    if det_result:
        total_input_tokens += det_result.input_tokens or 0
        total_output_tokens += det_result.output_tokens or 0

    # Step 2: Run N stochastic classifications at elevated temperature
    classifier.agent.temperature = stochastic_temperature

    individual_results = []

    # Run stochastic classifications sequentially to avoid rate-limit issues.
    # (All N runs share the same Agent/client; concurrent requests to Scaleway
    # may trigger throttling, producing degenerate 0-vote results.)
    # We request return_logprobs=True to capture token usage for cost tracking.
    for run_idx in range(n_runs):
        try:
            node_id, stoch_result = await classifier.classify_intent(
                user_text, context=context, return_logprobs=True
            )
            code = NODE_TO_CODE_SAFE(node_id)
            individual_results.append((node_id, code))
            if stoch_result:
                total_input_tokens += stoch_result.input_tokens or 0
                total_output_tokens += stoch_result.output_tokens or 0
        except Exception as e:
            logger.warning(f"Stochastic run {run_idx+1}/{n_runs} failed: {e}")

    # Restore original temperature
    classifier.agent.temperature = original_temp
    total_latency_ms = (time.perf_counter() - t_start) * 1000

    if not individual_results:
        logger.error("All stochastic runs failed")
        return ConsistencyResult(
            node_id=det_node_id,
            code=NODE_TO_CODE_SAFE(det_node_id),
            agreement_ratio=0.0,
            n_runs=n_runs,
            vote_distribution={},
            individual_results=[],
            is_confident=False,
            deterministic_result=det_result,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_latency_ms=total_latency_ms,
        )

    # Step 3: Compute majority vote and agreement
    node_counts = Counter(r[0] for r in individual_results)
    majority_node_id, majority_count = node_counts.most_common(1)[0]
    agreement_ratio = majority_count / len(individual_results)

    code_counts = Counter(r[1] for r in individual_results)
    vote_distribution = dict(code_counts.most_common())

    is_confident = agreement_ratio >= confidence_threshold

    majority_code = NODE_TO_CODE_SAFE(majority_node_id)

    logger.info(
        f"Semantic consistency for '{user_text[:40]}...': "
        f"{agreement_ratio:.0%} agreement ({majority_code}), "
        f"distribution: {vote_distribution}"
    )

    return ConsistencyResult(
        node_id=majority_node_id,
        code=majority_code,
        agreement_ratio=agreement_ratio,
        n_runs=len(individual_results),
        vote_distribution=vote_distribution,
        individual_results=individual_results,
        is_confident=is_confident,
        deterministic_result=det_result,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_latency_ms=total_latency_ms,
    )


def NODE_TO_CODE_SAFE(node_id: str) -> str:
    """Safely convert a node_id to its code, returning node_id if not found."""
    from .classifier import NODE_TO_CODE
    return NODE_TO_CODE.get(node_id, node_id)


async def evaluate_batch_consistency(
    classifier: ClassifierProtocol,
    test_cases: List[dict],
    n_runs: int = 5,
    stochastic_temperature: float = 0.7,
    confidence_threshold: float = 0.8,
) -> List[ConsistencyResult]:
    """
    Run semantic consistency evaluation on a batch of test cases.

    Args:
        classifier: Any classifier implementing classify_intent()
        test_cases: List of dicts with "input" and optional "context" keys
        n_runs: Number of stochastic runs per test case
        stochastic_temperature: Temperature for stochastic runs
        confidence_threshold: Agreement ratio threshold

    Returns:
        List of ConsistencyResult objects
    """
    results = []
    for i, test in enumerate(test_cases):
        user_text = test.get("input", "")
        context = test.get("context") or test.get("nightmare")
        logger.info(f"Consistency eval [{i+1}/{len(test_cases)}]: '{user_text[:40]}...'")

        result = await evaluate_semantic_consistency(
            classifier=classifier,
            user_text=user_text,
            context=context,
            n_runs=n_runs,
            stochastic_temperature=stochastic_temperature,
            confidence_threshold=confidence_threshold,
        )
        results.append(result)

    return results
