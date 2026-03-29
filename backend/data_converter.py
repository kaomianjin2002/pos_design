from __future__ import annotations

import argparse
from pathlib import Path

from utils import convert_to_conll, detect_language


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将中英文词性语料转换为 CoNLL 标准格式。")
    parser.add_argument("--input", required=True, help="原始语料文件路径")
    parser.add_argument("--output", required=True, help="输出 CoNLL 文件路径")
    parser.add_argument("--lang", choices=["zh", "en"], help="可选的语言参数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    lang = args.lang or detect_language(input_path.read_text(encoding="utf-8"))
    output_path = convert_to_conll(input_path, args.output, lang=lang)
    print(f"转换完成：{input_path} -> {output_path}，语言={lang}")


if __name__ == "__main__":
    main()
