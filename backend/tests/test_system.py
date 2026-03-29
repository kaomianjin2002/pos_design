import json
import threading
import time
from pathlib import Path
from urllib.request import Request, urlopen

from backend.main import run
from backend.utils import convert_to_conll, load_conll_sentences


def test_convert_to_conll(tmp_path: Path):
    raw = tmp_path / "sample.txt"
    output = tmp_path / "sample.conll"
    raw.write_text("中国/nr 政府/n 宣布/v\n", encoding="utf-8")
    convert_to_conll(raw, output, lang="zh")
    content = output.read_text(encoding="utf-8")
    assert "中国" in content
    assert "nr" in content


def test_load_conll_sentences():
    path = Path("backend/data/processed/chinese_sample.conll")
    sentences = load_conll_sentences(path)
    assert sentences
    assert sentences[0][0]["form"] == "中国"


def test_train_predict_stats_flow():
    port = 8765
    server_thread = threading.Thread(target=run, kwargs={"host": "127.0.0.1", "port": port}, daemon=True)
    server_thread.start()
    for _ in range(20):
        try:
            with urlopen(f"http://127.0.0.1:{port}/health") as response:
                if response.status == 200:
                    break
        except Exception:
            time.sleep(0.1)

    train_request = Request(
        f"http://127.0.0.1:{port}/train",
        data=json.dumps({"iterations": 3}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(train_request) as response:
        stats = json.loads(response.read().decode("utf-8"))
    assert "test_accuracy" in stats

    predict_request = Request(
        f"http://127.0.0.1:{port}/predict",
        data=json.dumps({"text": "The cat sits quietly", "tokenizer": "whitespace"}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(predict_request) as response:
        payload = json.loads(response.read().decode("utf-8"))
    assert len(payload["tokens"]) == 4
    assert payload["tokens"][0]["tag_name"]

    with urlopen(f"http://127.0.0.1:{port}/stats") as response:
        stats_payload = json.loads(response.read().decode("utf-8"))
    assert stats_payload["vocab_size"] >= 1


def test_english_ptb_tags_are_preserved(tmp_path: Path):
    sample = tmp_path / "fine.conll"
    sample.write_text(
        "1\tModels\t_\tNNS\tNNS\t_\t_\t_\t_\t_\n"
        "2\timproves\t_\tVBZ\tVBZ\t_\t_\t_\t_\t_\n"
        "3\tquickly\t_\tRB\tRB\t_\t_\t_\t_\t_\n",
        encoding="utf-8",
    )
    sentences = load_conll_sentences(sample)
    tags = [token["tag"] for token in sentences[0]]
    assert tags == ["NNS", "VBZ", "RB"]


def test_people_daily_bracket_format_conversion(tmp_path: Path):
    raw = tmp_path / "people_daily.txt"
    out = tmp_path / "people_daily.conll"
    raw.write_text("[北京/ns 大学/n]nt 是/v 名校/n 。/w\n", encoding="utf-8")
    convert_to_conll(raw, out, lang="zh")
    content = out.read_text(encoding="utf-8")
    assert "北京" in content
    assert "大学" in content
    assert "	ns	" in content
    assert "	n	" in content
    assert "n]" not in content
