"""Deterministic, strictly bounded NEWS-only multinomial Naive Bayes model.

Serialized token frequencies are PRIVATE research material, not public data.
No stock-price data, human review notes, or TEST gold enter this model.
"""
from __future__ import annotations

import math
from collections import Counter

LABELS = ("IMPORTANT", "UNIMPORTANT")
MAX_VOCABULARY = 100000
MAX_COUNT = 10000000


class NewsImportanceModel:
    def __init__(self, *, alpha: float, documents: dict[str, int],
                 counts: dict[str, dict[str, int]]) -> None:
        if type(alpha) not in (float, int) or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("Positive finite smoothing alpha required")
        if set(documents) != set(LABELS) or set(counts) != set(LABELS):
            raise ValueError("Only two news importance classes allowed")
        if any(type(documents[label]) is not int or not 1 <= documents[label] <= MAX_COUNT
               for label in LABELS):
            raise ValueError("Positive bounded training documents per class required")
        vocabulary: set[str] = set()
        for label in LABELS:
            row = counts[label]
            if not isinstance(row, dict) or len(row) > MAX_VOCABULARY:
                raise ValueError("Invalid vocabulary")
            for token, amount in row.items():
                if not isinstance(token, str) or not 2 <= len(token) <= 128 or (
                    type(amount) is not int or not 1 <= amount <= MAX_COUNT
                ):
                    raise ValueError("Invalid news token or count")
                vocabulary.add(token)
        if len(vocabulary) > MAX_VOCABULARY or not vocabulary:
            raise ValueError("Nonempty bounded news vocabulary required")
        self.alpha = float(alpha)
        self.documents = dict(documents)
        self.counts = {label: dict(counts[label]) for label in LABELS}
        self.vocabulary = vocabulary
        self.totals = {
            label: sum(self.counts[label].values()) + self.alpha * len(vocabulary)
            for label in LABELS
        }

    @classmethod
    def from_training(
        cls, train: list[tuple[list[str], str]], alpha: float,
    ) -> NewsImportanceModel:
        counts: dict[str, Counter[str]] = {label: Counter() for label in LABELS}
        documents: Counter[str] = Counter()
        for words, label in train:
            if label not in LABELS:
                raise ValueError("Only human binary news labels may be trained")
            counts[label].update(words)
            documents[label] += 1
        return cls(
            alpha=alpha, documents=dict(documents),
            counts={label: dict(counts[label]) for label in LABELS},
        )

    def predict(self, words: list[str]) -> dict[str, float]:
        frequencies = Counter(word for word in words if word in self.vocabulary)
        total_documents = sum(self.documents.values())
        logits = {
            label: math.log(
                (self.documents[label] + self.alpha) /
                (total_documents + self.alpha * len(LABELS))
            ) + sum(
                amount * math.log(
                    (self.counts[label].get(word, 0) + self.alpha) /
                    self.totals[label]
                ) for word, amount in frequencies.items()
            )
            for label in LABELS
        }
        peak = max(logits.values())
        exponentials = {
            label: math.exp(logits[label] - peak) for label in LABELS
        }
        normalizer = sum(exponentials.values())
        return {
            "IMPORTANT": exponentials["IMPORTANT"] / normalizer,
            "UNIMPORTANT": exponentials["UNIMPORTANT"] / normalizer,
            "UNCERTAIN": 0.0,
        }

    def to_state(self) -> dict[str, object]:
        return {
            "alpha": self.alpha,
            "documents": {key: self.documents[key] for key in LABELS},
            "counts": {
                label: dict(sorted(self.counts[label].items()))
                for label in LABELS
            },
        }

    @classmethod
    def from_state(cls, state: object) -> NewsImportanceModel:
        if not isinstance(state, dict) or set(state) != {
            "alpha", "documents", "counts",
        }:
            raise ValueError("Unsupported news model state schema")
        return cls(
            alpha=state["alpha"],
            documents=state["documents"],
            counts=state["counts"],
        )
