from __future__ import annotations

import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

try:
    from conllu import parse as parse_conllu
except ImportError:  # pragma: no cover - graceful fallback for environments without deps
    parse_conllu = None

CONLL_COLUMNS = [
    "id",
    "form",
    "lemma",
    "upos",
    "xpos",
    "feats",
    "head",
    "deprel",
    "deps",
    "misc",
]

PTB_PATTERN = re.compile(r"\(([^()\s]+)\s+([^()]+?)\)")
CHINESE_CHAR_PATTERN = re.compile(r"[\u4e00-\u9fff]")


def detect_language(text: str) -> str:
    return "zh" if CHINESE_CHAR_PATTERN.search(text) else "en"


def normalize_token(token: str) -> str:
    if any(char.isdigit() for char in token):
        return "<NUM>"
    return token.lower() if token.isascii() else token


class DatasetSplit(dict):
    train: list
    valid: list
    test: list



def _parse_slash_tagged_line(line: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for chunk in line.strip().split():
        if "/" not in chunk:
            continue
        word, tag = chunk.rsplit("/", 1)
        word = word.strip()
        tag = tag.strip()
        if word and tag:
            pairs.append((word, tag))
    return pairs



def _parse_ptb_line(line: str) -> list[tuple[str, str]]:
    return [(word.strip(), tag.strip()) for tag, word in PTB_PATTERN.findall(line) if word.strip() and tag.strip()]



def _iter_sentences(text: str, lang: str | None = None) -> Iterable[list[tuple[str, str]]]:
    lang = lang or detect_language(text)
    buffer: list[tuple[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            if buffer:
                yield buffer
                buffer = []
            continue
        pairs = _parse_ptb_line(line) if line.startswith("(") else _parse_slash_tagged_line(line)
        if pairs:
            if line.startswith("("):
                yield pairs
            else:
                yield pairs
    if buffer:
        yield buffer



def sentence_to_conll(sentence: Sequence[tuple[str, str]]) -> str:
    rows = []
    for index, (word, tag) in enumerate(sentence, start=1):
        rows.append(f"{index}\t{word}\t_\t{tag}\t{tag}\t_\t_\t_\t_\t_")
    return "\n".join(rows)



def convert_to_conll(input_path: str | Path, output_path: str | Path, lang: str | None = None) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    text = input_path.read_text(encoding="utf-8")
    sentences = list(_iter_sentences(text, lang=lang))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = "\n\n".join(sentence_to_conll(sentence) for sentence in sentences)
    if serialized:
        serialized += "\n"
    output_path.write_text(serialized, encoding="utf-8")
    return output_path



def _fallback_parse_conll(text: str) -> list[list[dict[str, str]]]:
    sentences: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                sentences.append(current)
                current = []
            continue
        if line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        padded = parts + ["_"] * (10 - len(parts))
        row = dict(zip(CONLL_COLUMNS, padded[:10]))
        current.append(row)
    if current:
        sentences.append(current)
    return sentences



def load_conll_sentences(path: str | Path, min_freq: int = 1) -> list[list[dict[str, str]]]:
    text = Path(path).read_text(encoding="utf-8")
    if parse_conllu is not None:
        parsed = parse_conllu(text)
        sentences = []
        for sentence in parsed:
            rows = []
            for token in sentence:
                rows.append(
                    {
                        "id": str(token.get("id", "_")),
                        "form": token.get("form") or "_",
                        "lemma": token.get("lemma") or "_",
                        "upos": token.get("upos") or token.get("xpos") or "X",
                        "xpos": token.get("xpos") or token.get("upos") or "X",
                        "feats": str(token.get("feats") or "_"),
                        "head": str(token.get("head") or "_"),
                        "deprel": str(token.get("deprel") or "_"),
                        "deps": str(token.get("deps") or "_"),
                        "misc": str(token.get("misc") or "_"),
                    }
                )
            sentences.append(rows)
    else:
        sentences = _fallback_parse_conll(text)

    token_counter = Counter(token["form"] for sentence in sentences for token in sentence)
    for sentence in sentences:
        for token in sentence:
            token["normalized"] = normalize_token(token["form"])
            token["normalized_rare"] = token["normalized"] if token_counter[token["form"]] >= min_freq else "<UNK>"
            token["tag"] = token["upos"] if token["upos"] != "_" else token["xpos"]
    return sentences



def split_dataset(sentences: Sequence[list[dict[str, str]]], seed: int = 42) -> DatasetSplit:
    items = list(sentences)
    rng = random.Random(seed)
    rng.shuffle(items)
    total = len(items)
    train_end = max(1, int(total * 0.8)) if total else 0
    valid_end = train_end + max(1, int(total * 0.1)) if total >= 3 else min(total, train_end)
    valid_end = min(valid_end, total)
    train = items[:train_end]
    valid = items[train_end:valid_end]
    test = items[valid_end:]
    if total >= 3 and not test:
        test = [train.pop()]
    return DatasetSplit(train=train, valid=valid, test=test)



def build_vocabulary(sentences: Sequence[list[dict[str, str]]]) -> dict[str, int]:
    counter = Counter(token["normalized_rare"] for sentence in sentences for token in sentence)
    vocab = {token: idx for idx, (token, _) in enumerate(counter.most_common(), start=1)}
    vocab["<PAD>"] = 0
    return vocab



def build_tagset(sentences: Sequence[list[dict[str, str]]]) -> list[str]:
    tags = sorted({token["tag"] for sentence in sentences for token in sentence})
    return tags



def sentence_words(sentence: Sequence[dict[str, str]]) -> list[str]:
    return [token["form"] for token in sentence]



def sentence_tags(sentence: Sequence[dict[str, str]]) -> list[str]:
    return [token["tag"] for token in sentence]
