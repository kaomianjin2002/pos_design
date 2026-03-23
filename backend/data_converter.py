from __future__ import annotations

import argparse
from pathlib import Path

from utils import convert_to_conll, detect_language


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Chinese/English POS corpora to CoNLL format.")
    parser.add_argument("--input", required=True, help="Path to the raw corpus file")
    parser.add_argument("--output", required=True, help="Path to the output CoNLL file")
    parser.add_argument("--lang", choices=["zh", "en"], help="Optional language override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    lang = args.lang or detect_language(input_path.read_text(encoding="utf-8"))
    output_path = convert_to_conll(input_path, args.output, lang=lang)
    print(f"Converted {input_path} -> {output_path} ({lang})")


if __name__ == "__main__":
    main()
