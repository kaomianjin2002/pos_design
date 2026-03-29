from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from .utils import detect_language, normalize_token


@dataclass
class FeatureExtractor:
    def extract(
        self,
        words: Sequence[str],
        index: int,
        prev_tag: str,
        tag: str,
    ) -> list[str]:
        word = words[index]
        normalized = normalize_token(word)
        prev_word = "<BOS>" if index == 0 else normalize_token(words[index - 1])
        next_word = "<EOS>" if index == len(words) - 1 else normalize_token(words[index + 1])
        features = [
            f"bias::{tag}",
            f"w0::{normalized}::{tag}",
            f"w-1::{prev_word}::{tag}",
            f"w+1::{next_word}::{tag}",
            f"trans::{prev_tag}->{tag}",
        ]
        if word.isascii():
            lower = word.lower()
            for length in (1, 2, 3):
                features.append(f"pref{length}::{lower[:length]}::{tag}")
                features.append(f"suff{length}::{lower[-length:]}::{tag}")
            features.append(f"title::{word[:1].istitle()}::{tag}")
            features.append(f"digit::{any(ch.isdigit() for ch in word)}::{tag}")
            features.append(f"upper::{word.isupper()}::{tag}")
        else:
            language = detect_language(word)
            features.append(f"lang::{language}::{tag}")
            features.append(f"char_len::{len(word)}::{tag}")
        return features


@dataclass
class StructuredPerceptron:
    tags: list[str]
    iterations: int = 8
    extractor: FeatureExtractor = field(default_factory=FeatureExtractor)
    weights: defaultdict[str, float] = field(default_factory=lambda: defaultdict(float))
    totals: defaultdict[str, float] = field(default_factory=lambda: defaultdict(float))
    timestamps: defaultdict[str, int] = field(default_factory=lambda: defaultdict(int))
    step: int = 0

    def _score(self, words: Sequence[str], index: int, prev_tag: str, tag: str) -> float:
        return sum(self.weights[feature] for feature in self.extractor.extract(words, index, prev_tag, tag))

    def viterbi_decode(self, words: Sequence[str]) -> list[str]:
        if not words:
            return []
        dp: list[dict[str, float]] = []
        backpointers: list[dict[str, str]] = []
        for index in range(len(words)):
            state_scores: dict[str, float] = {}
            state_backpointers: dict[str, str] = {}
            for tag in self.tags:
                best_score = None
                best_prev = None
                prev_candidates = ["<START>"] if index == 0 else self.tags
                for prev_tag in prev_candidates:
                    prev_score = 0.0 if index == 0 else dp[index - 1][prev_tag]
                    score = prev_score + self._score(words, index, prev_tag, tag)
                    if best_score is None or score > best_score:
                        best_score = score
                        best_prev = prev_tag
                state_scores[tag] = best_score if best_score is not None else float("-inf")
                state_backpointers[tag] = best_prev or "<START>"
            dp.append(state_scores)
            backpointers.append(state_backpointers)
        best_last = max(dp[-1], key=dp[-1].get)
        sequence = [best_last]
        for index in range(len(words) - 1, 0, -1):
            sequence.append(backpointers[index][sequence[-1]])
        sequence.reverse()
        return sequence

    def _update_feature(self, feature: str, value: float) -> None:
        elapsed = self.step - self.timestamps[feature]
        self.totals[feature] += elapsed * self.weights[feature]
        self.timestamps[feature] = self.step
        self.weights[feature] += value

    def update(self, words: Sequence[str], gold_tags: Sequence[str], pred_tags: Sequence[str]) -> None:
        prev_gold = "<START>"
        prev_pred = "<START>"
        for index in range(len(words)):
            self.step += 1
            if gold_tags[index] == pred_tags[index]:
                prev_gold = gold_tags[index]
                prev_pred = pred_tags[index]
                continue
            gold_features = self.extractor.extract(words, index, prev_gold, gold_tags[index])
            pred_features = self.extractor.extract(words, index, prev_pred, pred_tags[index])
            for feature in gold_features:
                self._update_feature(feature, 1.0)
            for feature in pred_features:
                self._update_feature(feature, -1.0)
            prev_gold = gold_tags[index]
            prev_pred = pred_tags[index]

    def finalize(self) -> None:
        for feature in list(self.weights.keys()):
            elapsed = self.step - self.timestamps[feature]
            self.totals[feature] += elapsed * self.weights[feature]
            averaged = self.totals[feature] / max(1, self.step)
            self.weights[feature] = averaged

    def fit(self, dataset: Sequence[tuple[list[str], list[str]]]) -> dict[str, list[float] | int]:
        history: list[float] = []
        for _ in range(self.iterations):
            correct = 0
            total = 0
            for words, tags in dataset:
                predicted = self.viterbi_decode(words)
                if predicted != tags:
                    self.update(words, tags, predicted)
                for gold, pred in zip(tags, predicted):
                    correct += int(gold == pred)
                    total += 1
            history.append(correct / total if total else 0.0)
        self.finalize()
        return {"iterations": self.iterations, "accuracy_history": history}

    def predict(self, words: Sequence[str]) -> list[str]:
        return self.viterbi_decode(words)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"tags": self.tags, "iterations": self.iterations, "weights": dict(self.weights)}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "StructuredPerceptron":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(tags=payload["tags"], iterations=payload.get("iterations", 8))
        model.weights.update(payload.get("weights", {}))
        return model


@dataclass
class MostFrequentTagBaseline:
    global_default: str = "NN"
    word_to_tag: dict[str, str] = field(default_factory=dict)

    def fit(self, dataset: Iterable[tuple[list[str], list[str]]]) -> None:
        counter: dict[str, Counter[str]] = defaultdict(Counter)
        overall = Counter()
        for words, tags in dataset:
            for word, tag in zip(words, tags):
                normalized = normalize_token(word)
                counter[normalized][tag] += 1
                overall[tag] += 1
        self.word_to_tag = {word: tag_counts.most_common(1)[0][0] for word, tag_counts in counter.items()}
        if overall:
            self.global_default = overall.most_common(1)[0][0]

    def predict(self, words: Sequence[str]) -> list[str]:
        return [self.word_to_tag.get(normalize_token(word), self.global_default) for word in words]
