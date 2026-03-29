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

TAG_DESCRIPTION_ZH = {
    # PKU 核心实词
    "n": "名词",
    "nr": "人名",
    "ns": "地名",
    "nt": "机构团体",
    "v": "动词",
    "vn": "名动词",
    "a": "形容词",
    "ad": "副形词",

    # PKU 常用虚词与辅助词
    "u": "助词",
    "p": "介词",
    "c": "连词",
    "d": "副词",

    # PKU 其他高频标注
    "r": "代词",
    "m": "数词",
    "q": "量词",
    "w": "标点符号",
    "x": "非汉字字符串",

    # 兼容常见扩展与历史写法
    "NN": "普通名词",
    "NR": "专有名词",
    "NT": "时间名词",
    "VV": "其他动词",
    "VC": "系动词",
    "VE": "有字句动词",
    "VA": "表语形容词",
    "JJ": "名词修饰语",
    "AD": "副词",
    "PN": "代词",
    "P": "介词",
    "CC": "并列连词",
    "CD": "数词",
    "M": "量词",
    "DT": "限定词",
    "DEC": "的（从句）",
    "DEG": "的（结构助词）",
    "DER": "得",
    "DEV": "地",
    "AS": "体标记",
    "MSP": "其他小品词",
    "SP": "句末语气词",
    "LC": "方位词",
    "FW": "外来词",
    "ON": "拟声词",
    "PU": "标点",

    # Penn Treebank 常见标签
    "NN": "单数名词",
    "NNS": "复数名词",
    "NNP": "单数专有名词",
    "NNPS": "复数专有名词",
    "VB": "动词原形",
    "VBP": "非第三人称单数现在时动词",
    "VBZ": "第三人称单数现在时动词",
    "VBD": "过去式动词",
    "VBG": "动名词/现在分词",
    "VBN": "过去分词",
    "JJ": "形容词",
    "JJR": "形容词比较级",
    "JJS": "形容词最高级",
    "RB": "副词",
    "RBR": "副词比较级",
    "RBS": "副词最高级",
    "PRP": "人称代词",
    "PRP$": "物主代词",
    "WP": "疑问代词",
    "WP$": "疑问物主代词",
    "IN": "介词/从属连词",
    "TO": "to",
    "CC": "并列连词",
    "DT": "限定词",
    "PDT": "前限定词",
    "WDT": "疑问限定词",
    "CD": "数词",
    "UH": "叹词",
    "SYM": "符号",
}


def normalize_pos_tag(tag: str) -> str:
    """训练标签保持原始标注集（PKU/PTB），不做粗粒度归并。"""
    if not tag:
        return "未知"
    return tag


def describe_tag_zh(tag: str) -> str:
    if not tag:
        return "未知"
    if CHINESE_CHAR_PATTERN.search(tag):
        return tag
    return TAG_DESCRIPTION_ZH.get(tag, f"未映射词性（{tag}）")

MODEL_NAMES_ZH = {
    "structured_perceptron": "结构化平均感知机",
    "most_frequent_baseline": "最高频词性基线",
}


def tag_to_chinese(tag: str) -> str:
    return describe_tag_zh(tag)


def tags_to_chinese(tags: Sequence[str]) -> list[str]:
    return [tag_to_chinese(tag) for tag in tags]


def distribution_to_chinese(distribution: dict[str, int]) -> dict[str, int]:
    return {tag_to_chinese(tag): value for tag, value in distribution.items()}


def confusion_to_chinese(confusion: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    return {
        tag_to_chinese(gold): {tag_to_chinese(pred): count for pred, count in row.items()}
        for gold, row in confusion.items()
    }


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



def _clean_people_daily_chunk(word: str, tag: str) -> tuple[str, str] | None:
    """清洗人民日报风格 chunk，例如 [北京/ns、大学/n]nt。"""
    word = word.strip().lstrip("[")
    tag = tag.strip()
    if "]" in tag:
        tag = tag.split("]", 1)[0]
    tag = tag.rstrip("]")
    if not word or not tag:
        return None
    if word.startswith("]"):
        return None
    return word, tag



def _parse_slash_tagged_line(line: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for chunk in line.strip().split():
        if "/" not in chunk:
            # 例如 ]nt 这种短语闭合标签，直接跳过
            continue
        word, tag = chunk.rsplit("/", 1)
        cleaned = _clean_people_daily_chunk(word, tag)
        if cleaned is not None:
            pairs.append(cleaned)
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
            raw_tag = token["upos"] if token["upos"] != "_" else token["xpos"]
            token["tag_raw"] = raw_tag
            token["tag"] = raw_tag
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
