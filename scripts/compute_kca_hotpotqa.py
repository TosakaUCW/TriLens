#!/usr/bin/env python3
"""KCA diagnostic on HotpotQA.

Pairing construction (HotpotQA has no native hallu field):
  * yes/no answers -> hallu = the opposite ("yes" <-> "no").
  * Span answers   -> hallu = a random wrong answer from the span pool
                      whose word count is within +/- 1 of the right answer.

Context is the concatenation of all paragraphs (both supporting and distractor),
matching the standard distractor-setting HotpotQA format.
"""
import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.data_utils import build_prompt, dtype_from_name, tokenize_text
from src.kca import LayerCaptures, compute_kca_features


YESNO = {"yes", "no"}


class RateColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        speed = task.speed
        if speed is None:
            return Text("-- rec/s")
        return Text(f"{speed:>6.2f} rec/s")


def build_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        RateColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("written={task.fields[written]} skipped={task.fields[skipped]}"),
        transient=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KCA diagnostic on HotpotQA.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    parser.add_argument(
        "--pairing",
        type=str,
        default="both",
        choices=["both", "random", "right", "hallucinated"],
    )
    parser.add_argument(
        "--pairing_strategy",
        type=str,
        default="distractor",
        choices=["distractor", "global_pool"],
        help=(
            "How to construct the hallucinated answer for span questions. "
            "'distractor' (default, harder): sample an entity from a distractor paragraph of the same record -- "
            "the entity is plausible because it is present in the provided context, just irrelevant to the question. "
            "'global_pool' (easier): sample a random answer from the full dataset answer pool; tends to produce "
            "semantically unrelated wrong answers that are trivially detectable."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_response_tokens", type=int, default=32)
    parser.add_argument("--max_context_paragraphs", type=int, default=10)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="KCA does not need attention weights, so sdpa (O(n) memory) is safe and much faster than eager on long contexts.",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--lens_dtype", type=str, default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--emit_dola", action="store_true", help="Also compute DoLa-style JSD(lens(x_l), lens(x_L)) per layer and include it in the output row.")

    parser.add_argument("--use_chat_template", action="store_true", default=True)
    parser.add_argument("--disable_chat_template", action="store_true")
    parser.add_argument("--system_prompt", type=str, default=None)

    return parser.parse_args()


def build_context(record: Dict[str, Any], max_paragraphs: int) -> str:
    parts = []
    for title, sents in record.get("context", [])[:max_paragraphs]:
        parts.append(f"{title}: {' '.join(sents)}")
    return "\n\n".join(parts)


def build_answer_pool(records: List[Dict[str, Any]]) -> Dict[int, List[str]]:
    """Group answers by word count. Used for length-matched wrong-answer sampling."""
    pool: Dict[int, List[str]] = defaultdict(list)
    seen = set()
    for r in records:
        a = str(r.get("answer", "")).strip()
        if not a or a.lower() in YESNO:
            continue
        if a in seen:
            continue
        seen.add(a)
        pool[len(a.split())].append(a)
    return pool


def sample_from_pool(
    gold: str, pool: Dict[int, List[str]], rng: random.Random
) -> Optional[str]:
    """Length-matched sample from the global answer pool."""
    n = len(gold.split())
    candidates: List[str] = []
    for delta in (0, 1, -1, 2, -2):
        bucket = pool.get(n + delta, [])
        candidates.extend(x for x in bucket if x.lower() != gold.lower())
        if candidates:
            break
    if not candidates:
        return None
    return rng.choice(candidates)


# Sequence of capitalized tokens; allows apostrophes and digits.
_CAP_TOKEN = r"[A-Z][a-zA-Z0-9'’.]*"
_ENTITY_RE = re.compile(rf"\b(?:{_CAP_TOKEN})(?:\s+(?:{_CAP_TOKEN}))*\b")


def extract_distractor_entities(
    record: Dict[str, Any], max_paragraphs: int
) -> List[str]:
    """Collect candidate entities from paragraphs NOT in supporting_facts.

    The pool contains:
      - the distractor paragraph titles themselves (guaranteed clean entities);
      - capitalized spans found inside distractor paragraph text.
    """
    supporting_titles: Set[str] = {
        str(t).strip() for t, _ in record.get("supporting_facts", []) or []
    }
    entities: List[str] = []
    seen: Set[str] = set()

    for title, sents in record.get("context", [])[:max_paragraphs]:
        title = str(title).strip()
        if not title or title in supporting_titles:
            continue
        # Title itself is a clean entity.
        if title.lower() not in seen:
            entities.append(title)
            seen.add(title.lower())
        # Capitalized spans within the distractor paragraph.
        for sent in sents:
            for m in _ENTITY_RE.finditer(str(sent)):
                span = m.group(0).strip().strip(".")
                # Drop 1-char tokens and sentence-initial single capitalised words
                # like "The", "In", "A" that _CAP_TOKEN happens to match.
                if len(span) < 2:
                    continue
                key = span.lower()
                if key in seen:
                    continue
                entities.append(span)
                seen.add(key)
    return entities


def sample_from_distractor(
    gold: str,
    distractor_entities: List[str],
    rng: random.Random,
) -> Optional[str]:
    """Length-matched sample from this record's distractor entities.

    Falls back to progressively wider length windows before giving up.
    """
    n = len(gold.split())
    gold_low = gold.lower()
    for delta in (0, 1, -1, 2, -2):
        cands = [
            e for e in distractor_entities
            if len(e.split()) == n + delta and e.lower() != gold_low
        ]
        if cands:
            return rng.choice(cands)
    return None


def make_hotpotqa_candidates(
    record: Dict[str, Any],
    pairing: str,
    pairing_strategy: str,
    rng: random.Random,
    answer_pool: Dict[int, List[str]],
    max_paragraphs: int,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Return (candidates, hallu_source) where hallu_source in
    {'yesno_flip', 'distractor_entity', 'global_pool', None}."""
    right = str(record.get("answer", "")).strip()
    if not right:
        return [], None

    question = str(record.get("question", ""))
    context = build_context(record, max_paragraphs)
    prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"

    if right.lower() in YESNO:
        hallu = "No" if right.lower() == "yes" else "Yes"
        hallu_source = "yesno_flip"
    else:
        hallu = None
        hallu_source = None
        if pairing_strategy == "distractor":
            distractor_entities = extract_distractor_entities(record, max_paragraphs)
            hallu = sample_from_distractor(right, distractor_entities, rng)
            if hallu is not None:
                hallu_source = "distractor_entity"
        if hallu is None:
            # Fallback to global pool (also covers pairing_strategy='global_pool').
            hallu = sample_from_pool(right, answer_pool, rng)
            if hallu is not None:
                hallu_source = "global_pool"
        if hallu is None:
            return [], None

    both = [
        {"prompt": prompt, "response": right, "label": 0, "response_type": "right"},
        {"prompt": prompt, "response": hallu, "label": 1, "response_type": "hallucinated"},
    ]

    if pairing == "both":
        return both, hallu_source
    if pairing == "random":
        return [both[rng.randint(0, 1)]], hallu_source
    if pairing == "right":
        return [both[0]], hallu_source
    if pairing == "hallucinated":
        return [both[1]], hallu_source
    raise ValueError(f"Unknown pairing: {pairing}")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    use_chat_template = args.use_chat_template and not args.disable_chat_template

    data_path = Path(args.data_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    dtype = dtype_from_name(args.dtype)
    lens_dtype = dtype_from_name(args.lens_dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map=None,
        attn_implementation=args.attn_implementation,
    ).to(args.device)
    model.eval()

    captures = LayerCaptures(model)

    lens_weight = model.lm_head.weight.detach().to(dtype=lens_dtype, device=args.device)
    lens_bias = (
        model.lm_head.bias.detach().to(dtype=lens_dtype, device=args.device)
        if model.lm_head.bias is not None
        else None
    )

    records = json.loads(data_path.read_text(encoding="utf-8"))
    # Build answer pool from the FULL dataset (for stable pairing), then slice records.
    answer_pool = build_answer_pool(records)

    if args.start_index:
        records = records[args.start_index:]
    if args.max_samples is not None:
        records = records[: args.max_samples]
    if not records:
        raise ValueError("No records to process")

    written = 0
    skipped = 0
    log_vocab = float(torch.log(torch.tensor(float(model.config.vocab_size))).item())

    meta = {
        "model_name_or_path": args.model_name_or_path,
        "num_layers": captures.n_layers,
        "vocab_size": int(model.config.vocab_size),
        "log_vocab_nats": log_vocab,
        "temperature": args.temperature,
        "lens_dtype": args.lens_dtype,
        "dataset": "hotpotqa",
    }

    try:
        with output_path.open("w", encoding="utf-8") as fout, build_progress() as progress:
            task_id = progress.add_task(
                "KCA HotpotQA",
                total=len(records),
                written=written,
                skipped=skipped,
            )

            for i, rec in enumerate(records):
                candidates, hallu_source = make_hotpotqa_candidates(
                    rec,
                    args.pairing,
                    args.pairing_strategy,
                    rng,
                    answer_pool,
                    args.max_context_paragraphs,
                )
                if not candidates:
                    skipped += 1
                    progress.update(task_id, advance=1, written=written, skipped=skipped)
                    continue

                for cand_idx, cand in enumerate(candidates):
                    prompt = cand["prompt"]
                    response = cand["response"]

                    full_prompt = build_prompt(
                        tokenizer,
                        prompt,
                        use_chat_template=use_chat_template,
                        system_prompt=args.system_prompt,
                    )
                    prompt_ids = tokenize_text(tokenizer, full_prompt)
                    response_ids = tokenize_text(tokenizer, response, max_len=args.max_response_tokens)

                    if response_ids.numel() == 0:
                        skipped += 1
                        continue

                    full_ids = torch.cat([prompt_ids, response_ids]).unsqueeze(0).to(args.device)
                    response_start = int(prompt_ids.numel())

                    feats = compute_kca_features(
                        model=model,
                        input_ids=full_ids,
                        response_start=response_start,
                        captures=captures,
                        temperature=args.temperature,
                        emit_dola=args.emit_dola,
                        lens_dtype=lens_dtype,
                        lens_weight=lens_weight,
                        lens_bias=lens_bias,
                    )

                    row = {
                        "index": i + args.start_index,
                        "candidate_index": cand_idx,
                        "task": "hotpotqa",
                        "pairing": args.pairing,
                        "pairing_strategy": args.pairing_strategy,
                        "hallu_source": hallu_source,
                        "response_type": cand.get("response_type"),
                        "label": cand.get("label"),
                        "num_layers": captures.n_layers,
                        "num_response_tokens": int(response_ids.numel()),
                        "core_positions": {
                            "user_prompt_start": 0,
                            "user_prompt_end": response_start,
                            "response_start": response_start,
                        },
                        "answer_type": "yesno" if str(rec.get("answer", "")).strip().lower() in YESNO else "span",
                        "H_a": feats["H_a"],
                        "H_m": feats["H_m"],
                        "H_x": feats["H_x"],
                        "H_pre": feats["H_pre"],
                        "H_p": feats["H_p"],
                        "JSD_am": feats["JSD_am"],
                        **({"JSD_to_final": feats["JSD_to_final"]} if "JSD_to_final" in feats else {}),
                        "meta": meta,
                    }
                    if "_id" in rec:
                        row["id"] = rec["_id"]

                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1

                progress.update(task_id, advance=1, written=written, skipped=skipped)

        print(f"Done. Wrote {written} rows (skipped {skipped}) to {output_path}.")
        print(f"log|V| in nats = {log_vocab:.4f}")
    finally:
        captures.remove()


if __name__ == "__main__":
    main()
