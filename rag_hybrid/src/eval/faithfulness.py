"""NLI-based verification of generated claims against cited retrieved evidence."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder as CrossEncoderType

# Keep the module importable without loading PyTorch until scoring is requested.
CrossEncoder: Any = None
_WORD_RE = re.compile(r"[\w]+", flags=re.UNICODE)
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "does", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "than", "this",
    "to", "was", "were", "what", "when", "where", "which", "who", "why", "with",
}

logger = logging.getLogger(__name__)


def _evidence_windows(text: str, claim: str) -> list[str]:
    """Split cited text into sentence and adjacent-sentence evidence windows."""
    normalized = re.sub(r"(?<=\w)-\s+(?=\w)", "", " ".join(text.split()))
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", normalized)
        if sentence.strip()
    ]
    if not sentences:
        return [normalized] if normalized else []

    windows = list(sentences)
    windows.extend(
        f"{sentences[index]} {sentences[index + 1]}"
        for index in range(len(sentences) - 1)
    )
    windows = list(dict.fromkeys(windows))
    claim_terms = {
        token.lower()
        for token in _WORD_RE.findall(claim)
        if len(token) > 2 and token.lower() not in _STOP_WORDS
    }
    if not claim_terms or len(windows) <= 4:
        return windows[:4]

    def relevance(window: str, order: int) -> tuple[float, int, int, int]:
        window_terms = {token.lower() for token in _WORD_RE.findall(window)}
        window_keys = {term[:6] if len(term) > 6 else term for term in window_terms}
        matched = sum(
            1
            for term in claim_terms
            if term in window_terms or (term[:6] if len(term) > 6 else term) in window_keys
        )
        coverage = matched / len(claim_terms)
        return coverage, matched, -len(window_terms), -order

    ranked = sorted(enumerate(windows), key=lambda item: relevance(item[1], item[0]), reverse=True)
    return [window for _, window in ranked[:4]]


class CitationVerificationResult(BaseModel):
    total_claims: int
    verified_claims: int
    faithfulness_score: float | None
    violations: list[str]


class FaithfulnessEvaluator:
    """Estimate support for each claim from the text of its cited chunk."""

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
        pair_claim_indexes: list[int] = []
        valid_indexes: set[int] = set()
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
                    f"Invalid citation [{citation}] or missing citation for claim: '{claim}'"
                )
            else:
                windows = _evidence_windows(str(evidence), claim)
                if not windows:
                    violations.append(
                        f"No readable evidence for citation [{citation}] and claim: '{claim}'"
                    )
                    continue
                valid_indexes.add(index)
                for window in windows:
                    valid_pairs.append((window, claim))
                    pair_claim_indexes.append(index)

        scores_by_index: dict[int, float] = {}
        if valid_pairs:
            predictions = self.nli.predict(valid_pairs, apply_softmax=True)
            prediction_rows = (
                [predictions]
                if getattr(predictions, "ndim", None) == 1
                else list(predictions)
            )
            for index, probabilities in zip(pair_claim_indexes, prediction_rows):
                if self.entail_idx >= len(probabilities):
                    logger.error("NLI entailment label index is outside the prediction vector")
                    entail_prob = 0.0
                else:
                    entail_prob = float(probabilities[self.entail_idx])
                scores_by_index[index] = max(scores_by_index.get(index, 0.0), entail_prob)
            if len(prediction_rows) != len(valid_pairs):
                logger.error("NLI prediction count did not match the number of evidence windows")

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
