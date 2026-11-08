"""LoRA SFT of Qwen3-8B (bf16) on the post-training chats (build_datasets.py: sft_train.jsonl, sft_eval.jsonl).

Each example is [system, user, assistant]. The prompt is rendered exactly as the deployed call renders it: the
model's chat template on the system and user turns with the generation prompt and thinking off (an empty think
block). The target is the assistant's JSON in the form the deployed grammar produces, then <|im_end|>: xgrammar
with disable_any_whitespace allows no free whitespace but keeps its default separators ", " and ": " (one space after
each comma and colon, as json.dumps writes by default; the untrained model's served answers read so), and the
schema's key order. A target written with (",", ":") is off the grammar: the served model is forced into ": " at its
third token and then rambles (lora_check.py, 2026-10-29). Prompt and target are tokenized apart, as the server
tokenizes the prompt and then generates; the loss is on the target only.

Before training it prints what the deployed call would see against what is trained (prompt tail, target, the
full-chat rendering, joint vs separate tokenization, the AWQ model's template) and the token lengths by act.
--generate N greedily decodes N held-out prompts per act after training and checks the JSON.

    python sft.py --model <dir> --train sft_train.jsonl --eval sft_eval.jsonl --out <adapter dir> --ckpt <scratch dir>
                  [--max-steps 50 --eval-limit 40 --check --generate 2]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import time
from typing import Any, Dict, List, Optional

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

KEYS = {"choose_procedure": ["reasoning", "statistic", "variance", "reference", "reason"],
        "spec_extraction": ["reasoning", "fields"]}
SHORT = {"choose_procedure": "choice", "spec_extraction": "spec"}


def as_deployed(content: str) -> str:
    """The assistant JSON as the deployed grammar writes it: separators ", " and ": ", no other whitespace."""
    return json.dumps(json.loads(content), ensure_ascii=False)


def prompt_of(tok: Any, messages: List[Dict[str, str]]) -> str:
    return tok.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True, enable_thinking=False)


def encode(tok: Any, row: Dict[str, Any], max_length: int) -> Optional[Dict[str, Any]]:
    prompt = tok(prompt_of(tok, row["messages"]), add_special_tokens=False)["input_ids"]
    target = tok(as_deployed(row["messages"][2]["content"]) + tok.eos_token, add_special_tokens=False)["input_ids"]
    if len(prompt) + len(target) > max_length:
        return None
    return {"input_ids": prompt + target, "labels": [-100] * len(prompt) + target, "act": row["meta"]["act"]}


def load(path: str, tok: Any, max_length: int, limit: Optional[int] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Encoded examples by act; examples over max_length are dropped (and counted), never truncated."""
    by_act: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    dropped: collections.Counter = collections.Counter()
    with open(path) as fh:
        for line in fh:
            row = json.loads(line)
            act = row["meta"]["act"]
            if limit is not None and len(by_act[act]) >= limit:
                continue
            ex = encode(tok, row, max_length)
            if ex is None:
                dropped[act] += 1
            else:
                by_act[act].append(ex)
    for act, rows in sorted(by_act.items()):
        n = [len(r["input_ids"]) for r in rows]
        t = [sum(1 for x in r["labels"] if x != -100) for r in rows]
        print(f"{os.path.basename(path)} {act}: {len(rows)} kept, {dropped[act]} over {max_length} dropped; "
              f"tokens min/median/max {min(n)}/{int(statistics.median(n))}/{max(n)}; "
              f"target median/max {int(statistics.median(t))}/{max(t)}", flush=True)
    return by_act


def collate(pad_id: int):
    def fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
        lab = torch.full((len(batch), n), -100, dtype=torch.long)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            ids[i, :k] = torch.tensor(b["input_ids"])
            lab[i, :k] = torch.tensor(b["labels"])
            att[i, :k] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}
    return fn


def check_format(tok: Any, path: str, awq: Optional[str]) -> None:
    """What the deployed call sees against what is trained, for the first example of each act."""
    seen = set()
    with open(path) as fh:
        for line in fh:
            row = json.loads(line)
            act = row["meta"]["act"]
            if act in seen:
                continue
            seen.add(act)
            prompt = prompt_of(tok, row["messages"])
            target = as_deployed(row["messages"][2]["content"])
            full = tok.apply_chat_template(row["messages"][:2] + [{"role": "assistant", "content": target}],
                                           tokenize=False, enable_thinking=False)
            sep = tok(prompt, add_special_tokens=False)["input_ids"] + tok(target + tok.eos_token, add_special_tokens=False)["input_ids"]
            joint = tok(prompt + target + tok.eos_token, add_special_tokens=False)["input_ids"]
            print(f"== {act}", flush=True)
            print("prompt tail:", repr(prompt[-90:]))
            print("target head:", repr(target[:160]))
            print("target tail:", repr(target[-80:]), "+", repr(tok.eos_token))
            print("full-chat rendering == prompt + target + eos + newline:", full == prompt + target + tok.eos_token + "\n")
            print("separate tokenization == joint tokenization:", sep == joint)
            print("keys in schema order:", list(json.loads(target)) == KEYS[act])
            print("separators ', ' and ': ' as the grammar writes them:", '", "' in target and '": "' in target
                  and '","' not in target.replace('\\"', ''))
            if len(seen) == len(KEYS):
                break
    if awq:
        other = AutoTokenizer.from_pretrained(awq)
        print("AWQ model's chat template identical:", other.chat_template == tok.chat_template)


def generate(model: Any, tok: Any, path: str, n: int) -> None:
    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    done: collections.Counter = collections.Counter()
    with open(path) as fh:
        for line in fh:
            row = json.loads(line)
            act = row["meta"]["act"]
            if done[act] >= n:
                continue
            done[act] += 1
            ids = tok(prompt_of(tok, row["messages"]), add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=900, do_sample=False)
            text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)
            body = text.split(tok.eos_token)[0]
            try:
                obj = json.loads(body)
                verdict = f"JSON ok, keys {list(obj) == KEYS[act]}, in the grammar's form {body == as_deployed(body)}"
                if act == "choose_procedure":
                    verdict += f"; answered {obj.get('statistic')}/{obj.get('variance')}|{obj.get('reference')}, best {row['meta'].get('best')}"
            except ValueError as e:
                verdict = f"not JSON ({e})"
            print(f"== generate {act} {row['meta'].get('task')} ({out.shape[1] - ids.shape[1]} tokens, {time.time() - t0:.0f} s): {verdict}")
            print(text[:1500], flush=True)
            if all(done[k] >= n for k in KEYS):
                break


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True, help="where the final adapter and the log go")
    ap.add_argument("--ckpt", required=True, help="checkpoint folder (scratch)")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=3072)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--eval-steps", type=int, default=50)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--eval-limit", type=int, help="held-out examples per act for the eval loss")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--check", action="store_true", help="kept for old commands; the format check always runs")
    ap.add_argument("--awq", help="the deployed AWQ model folder, to compare chat templates")
    ap.add_argument("--generate", type=int, default=0)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    check_format(tok, a.train, a.awq)
    train = load(a.train, tok, a.max_length)
    held = load(a.eval, tok, a.max_length, a.eval_limit)
    train_rows = [r for rows in train.values() for r in rows]
    evals = {SHORT[act]: rows for act, rows in held.items()}

    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    lora = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout, target_modules="all-linear",
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=a.ckpt, num_train_epochs=a.epochs, max_steps=a.max_steps, learning_rate=a.lr,
        lr_scheduler_type="cosine", warmup_steps=a.warmup, per_device_train_batch_size=a.batch,
        per_device_eval_batch_size=a.batch, gradient_accumulation_steps=a.accum, bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5, eval_strategy="steps", eval_steps=a.eval_steps,
        save_strategy="steps" if a.max_steps < 0 else "no", save_steps=a.save_steps, save_total_limit=2,
        save_only_model=True, report_to="none", remove_unused_columns=False, seed=a.seed,
        include_num_input_tokens_seen=True, dataloader_num_workers=0)
    trainer = Trainer(model=model, args=args, train_dataset=train_rows, eval_dataset=evals,
                      data_collator=collate(tok.pad_token_id))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = trainer.train()
    print(f"train: {result.metrics}; wall {time.time() - t0:.0f} s; "
          f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    print("final eval:", trainer.evaluate(), flush=True)
    trainer.save_model(a.out)
    with open(os.path.join(a.out, "log_history.json"), "w") as fh:
        json.dump({"args": vars(a), "log": trainer.state.log_history}, fh, indent=1)
    if a.generate:
        generate(trainer.model, tok, a.eval, a.generate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
