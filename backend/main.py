from __future__ import annotations

import json
from collections import Counter, defaultdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .core import MostFrequentTagBaseline, StructuredPerceptron
from .utils import build_tagset, build_vocabulary, load_conll_sentences, sentence_tags, sentence_words, split_dataset

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "backend" / "data" / "processed"
MODEL_DIR = ROOT / "backend" / "models"
FRONTEND_DIR = ROOT / "frontend"
MODEL_PATH = MODEL_DIR / "structured_perceptron.json"
STATS_PATH = MODEL_DIR / "training_stats.json"


class AppState:
    def __init__(self) -> None:
        self.model_path = MODEL_PATH
        self.stats_path = STATS_PATH
        self.model_dir = MODEL_DIR
        self.data_dir = DATA_DIR
        self.frontend_dir = FRONTEND_DIR


STATE = AppState()


class HTTPError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        self.status = status
        self.detail = detail
        super().__init__(detail)


class TrainRequest(dict):
    @property
    def dataset_paths(self) -> list[str] | None:
        return self.get("dataset_paths")

    @property
    def iterations(self) -> int:
        return int(self.get("iterations", 8))

    @property
    def seed(self) -> int:
        return int(self.get("seed", 42))


class PredictRequest(dict):
    @property
    def text(self) -> str:
        return str(self.get("text", ""))

    @property
    def tokenizer(self) -> str:
        return str(self.get("tokenizer", "auto"))



def get_default_datasets() -> list[Path]:
    return sorted(STATE.data_dir.glob("*.conll"))



def tokenize_text(text: str, tokenizer: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    if tokenizer == "whitespace":
        return stripped.split()
    if tokenizer == "char":
        return [char for char in stripped if not char.isspace()]
    if any("\u4e00" <= char <= "\u9fff" for char in stripped) and " " not in stripped:
        return [char for char in stripped if not char.isspace()]
    return stripped.split()



def compute_accuracy(gold: list[list[str]], pred: list[list[str]]) -> float:
    correct = total = 0
    for gold_sent, pred_sent in zip(gold, pred):
        for gold_tag, pred_tag in zip(gold_sent, pred_sent):
            correct += int(gold_tag == pred_tag)
            total += 1
    return correct / total if total else 0.0



def build_confusion_matrix(gold: list[list[str]], pred: list[list[str]]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for gold_sent, pred_sent in zip(gold, pred):
        for gold_tag, pred_tag in zip(gold_sent, pred_sent):
            matrix[gold_tag][pred_tag] += 1
    return {row: dict(cols) for row, cols in matrix.items()}



def tag_distribution(pred: list[list[str]]) -> dict[str, int]:
    counter = Counter(tag for sentence in pred for tag in sentence)
    return dict(counter.most_common())



def load_dataset(paths: list[Path], seed: int) -> dict[str, Any]:
    sentences = []
    for path in paths:
        sentences.extend(load_conll_sentences(path, min_freq=2))
    split = split_dataset(sentences, seed=seed)
    vocab = build_vocabulary(split["train"])
    tagset = build_tagset(split["train"])
    return {"sentences": sentences, "split": split, "vocab": vocab, "tagset": tagset}



def prepare_pairs(sentences: list[list[dict[str, str]]]) -> list[tuple[list[str], list[str]]]:
    return [(sentence_words(sentence), sentence_tags(sentence)) for sentence in sentences]



def evaluate_model(model: Any, dataset_pairs: list[tuple[list[str], list[str]]]) -> tuple[float, dict[str, dict[str, int]], dict[str, int]]:
    gold = [tags for _, tags in dataset_pairs]
    pred = [model.predict(words) for words, _ in dataset_pairs]
    return compute_accuracy(gold, pred), build_confusion_matrix(gold, pred), tag_distribution(pred)



def train_model(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    request = TrainRequest(payload or {})
    if request.iterations < 1 or request.iterations > 30:
        raise HTTPError(400, "iterations must be between 1 and 30")
    dataset_paths = [Path(path) for path in request.dataset_paths] if request.dataset_paths else get_default_datasets()
    if not dataset_paths:
        raise HTTPError(400, "No dataset files found for training.")

    data = load_dataset(dataset_paths, seed=request.seed)
    train_pairs = prepare_pairs(data["split"]["train"])
    valid_pairs = prepare_pairs(data["split"]["valid"])
    test_pairs = prepare_pairs(data["split"]["test"])
    if not train_pairs:
        raise HTTPError(400, "Training split is empty.")

    model = StructuredPerceptron(tags=data["tagset"], iterations=request.iterations)
    fit_summary = model.fit(train_pairs)
    baseline = MostFrequentTagBaseline()
    baseline.fit(train_pairs)

    valid_accuracy, _, _ = evaluate_model(model, valid_pairs) if valid_pairs else (0.0, {}, {})
    test_accuracy, confusion, distribution = evaluate_model(model, test_pairs) if test_pairs else (0.0, {}, {})
    baseline_accuracy, _, _ = evaluate_model(baseline, test_pairs) if test_pairs else (0.0, {}, {})

    model.save(STATE.model_path)
    stats = {
        "dataset_paths": [str(path) for path in dataset_paths],
        "train_sentences": len(train_pairs),
        "valid_sentences": len(valid_pairs),
        "test_sentences": len(test_pairs),
        "vocab_size": len(data["vocab"]),
        "tagset": data["tagset"],
        "fit_summary": fit_summary,
        "valid_accuracy": valid_accuracy,
        "test_accuracy": test_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "confusion_matrix": confusion,
        "tag_distribution": distribution,
    }
    STATE.model_dir.mkdir(parents=True, exist_ok=True)
    STATE.stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats



def predict_tags(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    request = PredictRequest(payload or {})
    if not STATE.model_path.exists():
        raise HTTPError(400, "Model not trained yet. Call /train first.")
    model = StructuredPerceptron.load(STATE.model_path)
    words = tokenize_text(request.text, request.tokenizer)
    if not words:
        raise HTTPError(400, "Input text is empty after tokenization.")
    tags = model.predict(words)
    return {"tokens": [{"word": word, "tag": tag} for word, tag in zip(words, tags)]}



def get_stats() -> dict[str, Any]:
    if not STATE.stats_path.exists():
        raise HTTPError(400, "Stats not available. Train the model first.")
    return json.loads(STATE.stats_path.read_text(encoding="utf-8"))


class PosTaggerHandler(BaseHTTPRequestHandler):
    server_version = "PosTaggerHTTP/1.0"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._serve_file(STATE.frontend_dir / "index.html", "text/html; charset=utf-8")
            return
        if parsed.path.startswith("/assets/"):
            relative = parsed.path.removeprefix("/assets/")
            target = (STATE.frontend_dir / relative).resolve()
            if not str(target).startswith(str(STATE.frontend_dir.resolve())) or not target.exists():
                self._json_response(HTTPStatus.NOT_FOUND, {"detail": "Asset not found"})
                return
            content_type = "text/plain; charset=utf-8"
            if target.suffix == ".js":
                content_type = "application/javascript; charset=utf-8"
            elif target.suffix == ".css":
                content_type = "text/css; charset=utf-8"
            self._serve_file(target, content_type)
            return
        if parsed.path == "/health":
            self._json_response(HTTPStatus.OK, {"status": "ok"})
            return
        if parsed.path == "/stats":
            try:
                self._json_response(HTTPStatus.OK, get_stats())
            except HTTPError as exc:
                self._json_response(exc.status, {"detail": exc.detail})
            return
        self._json_response(HTTPStatus.NOT_FOUND, {"detail": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(content_length).decode("utf-8") if content_length else "{}"
        try:
            payload = json.loads(raw or "{}")
        except json.JSONDecodeError:
            self._json_response(HTTPStatus.BAD_REQUEST, {"detail": "Invalid JSON payload"})
            return
        try:
            if parsed.path == "/train":
                self._json_response(HTTPStatus.OK, train_model(payload))
                return
            if parsed.path == "/predict":
                self._json_response(HTTPStatus.OK, predict_tags(payload))
                return
            self._json_response(HTTPStatus.NOT_FOUND, {"detail": "Not found"})
        except HTTPError as exc:
            self._json_response(exc.status, {"detail": exc.detail})

    def _serve_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self._json_response(HTTPStatus.NOT_FOUND, {"detail": "File not found"})
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _json_response(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return



def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), PosTaggerHandler)
    print(f"Serving POS tagger on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    run()
