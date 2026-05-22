from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch

NO_ANSWER_TEXT = "No answer."


def load_records(path: Path, split: Optional[str]) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            if split is not None:
                if split not in raw:
                    raise KeyError(f"split '{split}' not found in JSON keys: {list(raw.keys())}")
                if not isinstance(raw[split], list):
                    raise TypeError(f"JSON split '{split}' is not a list")
                return raw[split]
            for key in ("data", "train", "validation", "test", "dev"):
                if key in raw and isinstance(raw[key], list):
                    return raw[key]
            raise TypeError("JSON is a dict but no list-like split found; pass --split")
    except json.JSONDecodeError:
        pass

    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def pick_first_key(record: Dict[str, Any], candidates: Iterable[str], explicit_key: Optional[str]) -> Optional[str]:
    if explicit_key is not None:
        if explicit_key not in record:
            raise KeyError(f"Key '{explicit_key}' not found in record keys: {list(record.keys())}")
        return explicit_key
    for key in candidates:
        if key in record:
            return key
    return None


def build_prompt(tokenizer, prompt: str, use_chat_template: bool, system_prompt: Optional[str]) -> str:
    if not use_chat_template or not hasattr(tokenizer, "apply_chat_template"):
        return prompt
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def tokenize_text(tokenizer, text: str, max_len: Optional[int] = None) -> torch.Tensor:
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids = encoded["input_ids"][0]
    if max_len is not None:
        ids = ids[:max_len]
    return ids


def infer_halueval_task(record: Dict[str, Any], specified: str) -> str:
    if specified != "auto":
        return specified
    keys = set(record.keys())
    if {"user_query", "chatgpt_response"}.issubset(keys):
        return "general"
    if {"knowledge", "question", "right_answer", "hallucinated_answer"}.issubset(keys):
        return "qa"
    if {"knowledge", "dialogue_history", "right_response", "hallucinated_response"}.issubset(keys):
        return "dialogue"
    if {"document", "right_summary", "hallucinated_summary"}.issubset(keys):
        return "summarization"
    raise ValueError(f"Cannot infer task from keys: {sorted(keys)}")


def normalize_binary_label(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value > 0)
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "1", "hallucinated", "hallucination"}:
        return 1
    if text in {"no", "n", "false", "0", "non-hallucinated", "non_hallucinated"}:
        return 0
    return None


def make_halueval_candidates(
    record: Dict[str, Any],
    task: str,
    pairing: str,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if pairing == "auto":
        pairing = "single" if task == "general" else "random"

    if task == "general":
        prompt = str(record["user_query"])
        response = str(record["chatgpt_response"])
        label = normalize_binary_label(record.get("hallucination"))
        return [{"prompt": prompt, "response": response, "label": label, "response_type": "chatgpt_response"}]

    if task == "qa":
        prompt = f"Knowledge:\n{record['knowledge']}\n\nQuestion:\n{record['question']}\n\nAnswer:"
        right = str(record["right_answer"])
        hallu = str(record["hallucinated_answer"])
    elif task == "dialogue":
        prompt = f"Knowledge:\n{record['knowledge']}\n\nDialogue History:\n{record['dialogue_history']}\n\nAssistant Response:"
        right = str(record["right_response"])
        hallu = str(record["hallucinated_response"])
    elif task == "summarization":
        prompt = f"Document:\n{record['document']}\n\nSummary:"
        right = str(record["right_summary"])
        hallu = str(record["hallucinated_summary"])
    else:
        raise ValueError(f"Unsupported task: {task}")

    both = [
        {"prompt": prompt, "response": right, "label": 0, "response_type": "right"},
        {"prompt": prompt, "response": hallu, "label": 1, "response_type": "hallucinated"},
    ]

    if pairing == "both":
        return both
    if pairing == "random":
        return [both[rng.randint(0, 1)]]
    if pairing == "right":
        return [both[0]]
    if pairing == "hallucinated":
        return [both[1]]
    raise ValueError(f"Pairing '{pairing}' is invalid for task '{task}'")


def flatten_squad2(path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "data" not in raw or not isinstance(raw["data"], list):
        raise ValueError("SQuAD2.0 JSON must contain a 'data' list")

    rows: List[Dict[str, Any]] = []
    for article in raw["data"]:
        title = str(article.get("title", ""))
        for paragraph in article.get("paragraphs", []):
            context = str(paragraph.get("context", ""))
            for qa in paragraph.get("qas", []):
                answers = [a.get("text", "").strip() for a in qa.get("answers", []) if a.get("text", "").strip()]
                plausible = [
                    a.get("text", "").strip()
                    for a in qa.get("plausible_answers", [])
                    if a.get("text", "").strip()
                ]
                rows.append(
                    {
                        "id": str(qa["id"]),
                        "title": title,
                        "context": context,
                        "question": str(qa.get("question", "")),
                        "answers": answers,
                        "plausible_answers": plausible,
                        "is_impossible": bool(qa.get("is_impossible", False)),
                    }
                )
    if not rows:
        raise ValueError(f"No QA items found in {path}")
    return rows


def build_answer_pool(records: List[Dict[str, Any]]) -> List[str]:
    pool = set()
    for rec in records:
        for answer in rec.get("answers", []):
            if answer:
                pool.add(answer)
        for answer in rec.get("plausible_answers", []):
            if answer:
                pool.add(answer)
    pool.add(NO_ANSWER_TEXT)
    return list(pool)


def sample_wrong_answer(gold_answers: List[str], answer_pool: List[str], rng: random.Random) -> str:
    gold_set = set(gold_answers)
    candidates = [x for x in answer_pool if x not in gold_set]
    if not candidates:
        return NO_ANSWER_TEXT
    return rng.choice(candidates)


def make_squad2_candidates(
    record: Dict[str, Any],
    pairing: str,
    rng: random.Random,
    answer_pool: List[str],
) -> List[Dict[str, Any]]:
    if pairing in {"auto", "single"}:
        pairing = "random"

    prompt = f"Context:\n{record['context']}\n\nQuestion:\n{record['question']}\n\nAnswer:"
    is_impossible = bool(record.get("is_impossible", False))
    gold_answers = record.get("answers", [])
    plausible_answers = record.get("plausible_answers", [])

    if is_impossible:
        right = NO_ANSWER_TEXT
        hallu = plausible_answers[0] if plausible_answers else sample_wrong_answer([right], answer_pool, rng)
    else:
        if not gold_answers:
            return []
        right = gold_answers[0]
        hallu = plausible_answers[0] if plausible_answers else sample_wrong_answer(gold_answers, answer_pool, rng)

    both = [
        {"prompt": prompt, "response": right, "label": 0, "response_type": "right"},
        {"prompt": prompt, "response": hallu, "label": 1, "response_type": "hallucinated"},
    ]

    if pairing == "both":
        return both
    if pairing == "random":
        return [both[rng.randint(0, 1)]]
    if pairing == "right":
        return [both[0]]
    if pairing == "hallucinated":
        return [both[1]]
    raise ValueError(f"Pairing '{pairing}' is invalid for task 'squad2'")
