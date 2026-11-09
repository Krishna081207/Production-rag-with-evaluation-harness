"""NLI-based verification of generated claims against cited retrieved evidence."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder as CrossEncoderType

# Keep the module importable without loading PyTorch until scoring is requested.
CrossEncoder: Any = None

logger = logging.getLogger(__name__)


class CitationVerificationResult(BaseModel):
    total_claims: int
    verified_claims: int
    faithfulness_score: float | None
    violations: list[str]


class FaithfulnessEvaluator:
    """Estimate support for each claim from its cited chunk's parent context."""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small"):
        model_class = CrossEncoder
        if model_class is None:
            from sentence_transformers import CrossEncoder as model_class

        self.nli: CrossEncoderType = model_class(model_name)
        config = getattr(getattr(self.nli, "model", None), "config", None)
        label2id = getattr(config, "label2id", {}) or {}
        self.entail_idx = int(label2id.get("entailment", 1))

    def evaluate_generation(
        self,
        claims_with_citations: list[dict[str, str]],
        retrieved_chunks: Mapping[str, str],
        entail_thresh: float = 0.70,
    ) -> CitationVerificationResult:
        if not 0.0 <= entail_thresh <= 1.0:
            raise ValueError("entail_thresh must be between 0 and 1")
        if not claims_with_citations:
            return CitationVerificationResult(
                total_claims=0,
                verified_claims=0,
                faithfulness_score=None,
                violations=[],
            )

        valid_pairs: list[tuple[str, str]] = []
        valid_indexes: list[int] = []
        violations: list[str] = []
        for index, entry in enumerate(claims_with_citations):
            claim = entry.get("claim", "").strip()
            citation = entry.get("citation", "").strip()
            chunk_id = citation.removeprefix("Chunk_")
            evidence = retrieved_chunks.get(citation, retrieved_chunks.get(chunk_id))
            if not claim:
                violations.append("Unverifiable answer sentence: the claim was empty.")
            elif not citation or evidence is None:
                violations.append(
                    f"Invalid or missing citation [{citation}] for claim: '{claim}'"
                )
            else:
                valid_pairs.append((str(evidence), claim))
                valid_indexes.append(index)

        scores_by_index: dict[int, float] = {}
        if valid_pairs:
            predictions = self.nli.predict(valid_pairs, apply_softmax=True)
            prediction_rows = (
                [predictions]
                if getattr(predictions, "ndim", None) == 1
                else list(predictions)
            )
            for index, probabilities in zip(valid_indexes, prediction_rows):
                if self.entail_idx >= len(probabilities):
                    logger.error("NLI entailment label index is outside the prediction vector")
                    scores_by_index[index] = 0.0
                else:
                    scores_by_index[index] = float(probabilities[self.entail_idx])
            if len(prediction_rows) != len(valid_indexes):
                logger.error("NLI prediction count did not match the number of cited claims")

        verified = 0
        for index, entry in enumerate(claims_with_citations):
            if index not in scores_by_index:
                if index in valid_indexes:
                    violations.append(
                        f"Evidence check returned no score for claim: '{entry.get('claim', '').strip()}'"
                    )
                continue
            entail_prob = scores_by_index[index]
            if entail_prob >= entail_thresh:
                verified += 1
            else:
                claim = entry.get("claim", "").strip()
                violations.append(
                    f"Unsupported claim: '{claim}' (entailment probability={entail_prob:.2f})"
                )

        score = round(verified / len(claims_with_citations), 4)
        return CitationVerificationResult(
            total_claims=len(claims_with_citations),
            verified_claims=verified,
            faithfulness_score=score,
            violations=violations,
        )
