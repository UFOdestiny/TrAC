"""The evaluation matrix: benchmarks, backbones, and the cache tags of their settings."""

DATASETS = [
    ("gsm8k", "GSM8K"),
    ("math500", "MATH500"),
    ("minerva", "Minerva"),
    ("olympiad", "Olympiad"),
    ("aime3", "AIME"),
    ("gpqa_diamond", "GPQA"),
    ("mmlupro", "MMLU-Pro"),
    ("triviaqa", "TriviaQA"),
]
MODELS = [
    ("qwen8b", "Qwen3-8B"),
    ("llama8b", "Llama-3.1-8B"),
    ("ministral8b", "Ministral-8B"),
    ("gptoss20b", "gpt-oss-20b"),
]
DATASET_KEYS = [key for key, _ in DATASETS]
MODEL_KEYS = [key for key, _ in MODELS]
MATH_DATASETS = {"gsm8k", "math500", "minerva", "olympiad", "aime3"}
MIN_CLASS_COUNT = 10  # settings need this many correct and incorrect responses to enter aggregates

K = 8  # traces per question in the main matrix (the original trace and seven samples)
NEW_BACKBONE = "qwen32b"  # held-out larger backbone, one trace per question


def tag_of(dataset: str, model: str, k: int = K) -> str:
    """Cache tag of a setting."""
    return f"{dataset}_{model}_k{k}"


def parse_tag(tag: str) -> tuple[str, str]:
    """(dataset, model) of a cache tag."""
    dataset, model, _ = tag.rsplit("_", 2)
    return dataset, model
