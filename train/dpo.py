"""DPO of the SFT adapter on on-policy pairs (dpo_pairs.py: dpo_train.jsonl, dpo_eval.jsonl), one A40.

The policy is the SFT adapter itself, trained further; the reference is the SFT model, whose log probabilities of
every chosen and rejected answer are computed once before training (adapter as loaded, eval mode). The result is
again one LoRA adapter on the bf16 base, served by vLLM exactly as the SFT adapter. (TRL's DPOTrainer with a PEFT
model takes the base with the adapter switched off as its reference, which here would be the untrained model.)

Prompt and answers are tokenized as in sft.py: the deployed chat template with thinking off, each answer in the
grammar's form (sft.as_deployed) followed by <|im_end|>; pairs whose longer answer does not fit max_length are
dropped. Loss per pair: -log sigmoid(beta * ((log pi_c - log ref_c) - (log pi_r - log ref_r))) plus nll times the
mean negative log likelihood of the chosen answer's tokens (keeps the chosen answer's probability from falling with
the rejected one's; 0 turns it off). Log probabilities are sums over the answer's tokens.

    python dpo.py --model <base> --adapter <sft adapter> --train dpo_train.jsonl --eval dpo_eval.jsonl --out <dir>
                  --ckpt <scratch dir> [--max-steps 10 --limit-train 200 --eval-limit 20 --generate 2 --generate-file sft_eval.jsonl]
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import statistics
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from sft import SHORT, as_deployed, generate, prompt_of


def encode(tok: Any, row: Dict[str, Any], max_length: int) -> Optional[Dict[str, Any]]:
    p = tok(prompt_of(tok, row["prompt"]), add_special_tokens=False)["input_ids"]
    c, r = (tok(as_deployed(row[k][0]["content"]) + tok.eos_token, add_special_tokens=False)["input_ids"]
            for k in ("chosen", "rejected"))
    if len(p) + max(len(c), len(r)) > max_length:
        return None
    return {"p": p, "c": c, "r": r, "act": row["meta"]["act"]}


def load(path: str, tok: Any, max_length: int, limit: Optional[int], per_act: bool, seed: int) -> List[Dict[str, Any]]:
    """Encoded pairs; limit is per act for the held-out file, overall (a random subset) for the training file."""
    rows = [json.loads(l) for l in open(path) if l.strip()]
    random.Random(seed).shuffle(rows)
    out: List[Dict[str, Any]] = []
    kept: collections.Counter = collections.Counter()
    dropped: collections.Counter = collections.Counter()
    for row in rows:
        act = row["meta"]["act"]
        if limit is not None and (kept[act] if per_act else len(out)) >= limit:
            continue
        ex = encode(tok, row, max_length)
        if ex is None:
            dropped[act] += 1
            continue
        out.append(ex)
        kept[act] += 1
    for act in sorted(kept):
        n = [len(e["p"]) + max(len(e["c"]), len(e["r"])) for e in out if e["act"] == act]
        print(f"{os.path.basename(path)} {act}: {kept[act]} pairs kept, {dropped[act]} over {max_length} dropped; "
              f"longer side tokens median/max {int(statistics.median(n))}/{max(n)}", flush=True)
    return out


def seq_logp(model: Any, p: List[int], t: List[int]):
    """Sum and mean log probability of the answer tokens t after the prompt p (logits only at the answer)."""
    ids = torch.tensor([p + t], device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=ids, logits_to_keep=len(t) + 1).logits[0, :-1].float()
    lp = torch.log_softmax(logits, dim=-1).gather(1, torch.tensor(t, device="cuda")[:, None])[:, 0]
    return lp.sum(), lp.mean()


def pair_terms(model: Any, ex: Dict[str, Any], beta: float):
    lc, mc = seq_logp(model, ex["p"], ex["c"])
    lr, _ = seq_logp(model, ex["p"], ex["r"])
    dc, dr = lc - ex["ref_c"], lr - ex["ref_r"]
    margin = beta * (dc - dr)
    return -F.logsigmoid(margin), margin, dc, dr, mc


def evaluate(model: Any, rows: List[Dict[str, Any]], beta: float) -> Dict[str, float]:
    model.eval()
    acc: Dict[str, List[List[float]]] = collections.defaultdict(list)
    with torch.no_grad():
        for ex in rows:
            loss, margin, dc, dr, _ = pair_terms(model, ex, beta)
            acc[SHORT[ex["act"]]].append([loss.item(), float(margin.item() > 0), dc.item(), dr.item()])
    model.train()
    out = {}
    for k, v in acc.items():
        for j, name in enumerate(("loss", "accuracy", "chosen_logratio", "rejected_logratio")):
            out[f"eval_{k}_{name}"] = round(statistics.mean(x[j] for x in v), 4)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True, help="the SFT adapter: the starting policy and the reference")
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--accum", type=int, default=16, help="pairs per optimizer step")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--nll", type=float, default=0.2)
    ap.add_argument("--max-length", type=int, default=3072)
    ap.add_argument("--limit-train", type=int)
    ap.add_argument("--eval-limit", type=int, default=100, help="held-out pairs per act")
    ap.add_argument("--eval-steps", type=int, default=50)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--generate", type=int, default=0)
    ap.add_argument("--generate-file", help="an SFT-format file (sft_eval.jsonl) for --generate")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.ckpt, exist_ok=True)
    torch.manual_seed(a.seed)

    tok = AutoTokenizer.from_pretrained(a.model)
    train = load(a.train, tok, a.max_length, a.limit_train, False, a.seed)
    held = load(a.eval, tok, a.max_length, a.eval_limit, True, a.seed)

    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, a.adapter, is_trainable=True)
    model.print_trainable_parameters()

    # the reference: the SFT model as loaded
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        for ex in train + held:
            ex["ref_c"] = seq_logp(model, ex["p"], ex["c"])[0].item()
            ex["ref_r"] = seq_logp(model, ex["p"], ex["r"])[0].item()
    print(f"reference log probabilities for {len(train) + len(held)} pairs in {time.time() - t0:.0f} s; "
          f"chosen minus rejected, median over training pairs "
          f"{statistics.median(e['ref_c'] - e['ref_r'] for e in train):.2f}", flush=True)

    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train) / a.accum)
    total = a.max_steps if a.max_steps > 0 else math.ceil(a.epochs * steps_per_epoch)
    sched = get_cosine_schedule_with_warmup(opt, int(a.warmup * total), total)
    log: List[Dict[str, Any]] = [{"step": 0, **evaluate(model, held, a.beta)}]
    print(log[-1], flush=True)
    rng = random.Random(a.seed)
    order: List[int] = []
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    window: List[List[float]] = []
    for step in range(1, total + 1):
        for _ in range(a.accum):
            if not order:
                order = list(range(len(train)))
                rng.shuffle(order)
            ex = train[order.pop()]
            loss, margin, dc, dr, mc = pair_terms(model, ex, a.beta)
            ((loss - a.nll * mc) / a.accum).backward()
            window.append([loss.item(), float(margin.item() > 0), dc.item(), dr.item()])
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % 5 == 0 or step == total:
            m = {name: round(statistics.mean(x[j] for x in window), 4)
                 for j, name in enumerate(("loss", "accuracy", "chosen_logratio", "rejected_logratio"))}
            log.append({"step": step, "lr": sched.get_last_lr()[0], **m, "s_per_step": round((time.time() - t0) / step, 1)})
            print(log[-1], flush=True)
            window = []
        if step % a.eval_steps == 0 or step == total:
            log.append({"step": step, **evaluate(model, held, a.beta)})
            print(log[-1], flush=True)
        if a.max_steps < 0 and step % a.save_steps == 0:
            model.save_pretrained(os.path.join(a.ckpt, "last"))
    print(f"train: {total} steps of {a.accum} pairs in {time.time() - t0:.0f} s; peak GPU memory "
          f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    model.save_pretrained(a.out)
    with open(os.path.join(a.out, "log_history.json"), "w") as fh:
        json.dump({"args": vars(a), "log": log}, fh, indent=1)
    if a.generate and a.generate_file:
        generate(model, tok, a.generate_file, a.generate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
