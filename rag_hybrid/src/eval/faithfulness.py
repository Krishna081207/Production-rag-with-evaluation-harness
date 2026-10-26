# Faithfulness evaluation utilities for rag_hybrid
"""Evaluation harness that checks generated claims against retrieved evidence.

The evaluator expects a list of ``{"claim": <str>, "citation": <chunk_id>}`` objects
and a mapping from ``chunk_id`` to the parent text that was supplied to the generator.
It returns a deterministic ``CitationVerificationResult``.
"""

from typing import Dict, List

from pydantic import BaseModel
from sentence_transformers import CrossEncoder


class CitationVerificationResult(BaseModel):
    total_claims: int
    verified_claims: int
    faithfulness_score: float
    violations: List[str]


class FaithfulnessEvaluator:
    """NLI‑based verifier for RAG generation outputs."""

    def __init__(self):
        # Model returns logits for [Contradiction, Neutral, Entailment]
        self.nli = CrossEncoder("cross-encoder/nli-deberta-v3-small")
        self.entail_idx = 2

    def evaluate_generation(
        self,
        claims_with_citations: List[Dict[str, str]],
        retrieved_chunks: Dict[str, str],
        entail_thresh: float = 0.70,
    ) -> CitationVerificationResult:
        if not claims_with_citations:
            return CitationVerificationResult(
                total_claims=0,
                verified_claims=0,
                faithfulness_score=1.0,
                violations=[],
            )
        verified = 0
        violations: List[str] = []
        for entry in claims_with_citations:
            claim = entry["claim"]
            cid = entry["citation"]
            if cid not in retrieved_chunks:
                violations.append(
                    f"Invalid citation [{cid}] for claim: '{claim}' (not in retrieved context)"
                )
                continue
            evidence = retrieved_chunks[cid]
            probs = self.nli.predict([[evidence, claim]], apply_softmax=True)[0]
            entail_prob = probs[self.entail_idx]
            if entail_prob >= entail_thresh:
                verified += 1
            else:
                violations.append(
                    f"Hallucination: '{claim}' vs evidence (entail prob={entail_prob:.2f})"
                )
        score = round(verified / len(claims_with_citations), 4)
        return CitationVerificationResult(
            total_claims=len(claims_with_citations),
            verified_claims=verified,
            faithfulness_score=score,
            violations=violations,
        )
