"""Does vLLM apply a LoRA adapter as PEFT does? Same prompts, same adapter, two engines.

    lora_check.py hf   --model <base> --adapter <dir> --tasks <a.jsonl> <b.jsonl> --limit 3 --out hf.json
    lora_check.py vllm --model <base> --adapter <dir> --hf hf.json --schemas schemas.json --tag <name>

`hf` renders each prompt with the training chat template (thinking off), decodes greedily with transformers + PEFT and
scores its own output token by token (teacher forcing) with the adapter at its trained scale, at half scale and switched
off. `vllm` takes the same prompt token ids and (1) decodes greedily with the adapter, without and with the deployed
grammar, (2) scores the transformers output with prompt_logprobs, with and without the adapter, (3) checks that the
server's chat rendering gives the same prompt ids. A LoRA applied correctly agrees with PEFT to bf16 noise, as the base
model does; a scale error shows as agreement with the half-scale column.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

KEYS_ACT = ("choose_procedure", "spec_extraction")


def tasks_of(paths: List[str], limit: int) -> List[Dict[str, Any]]:
    out = []
    for p in paths:
        with open(p) as fh:
            for i, line in enumerate(fh):
                if i >= limit:
                    break
                t = json.loads(line)
                out.append({"id": t["id"], "act": t["act"], "messages": t["messages"][:2]})
    return out


def run_hf(a: argparse.Namespace) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, a.adapter).eval()

    def score(ids: List[int], n_prompt: int) -> Dict[str, List]:
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ids], device="cuda")).logits[0].float()
        lp = torch.log_softmax(logits[n_prompt - 1:-1], dim=-1)
        tgt = torch.tensor(ids[n_prompt:], device="cuda")
        top = lp.max(dim=-1)
        return {"lp": lp.gather(1, tgt[:, None])[:, 0].tolist(), "top_id": top.indices.tolist(), "top_lp": top.values.tolist()}

    def set_scale(f: float) -> None:
        for m in model.modules():
            if hasattr(m, "scaling") and isinstance(m.scaling, dict) and "default" in m.scaling:
                m.scaling["default"] = m.lora_alpha["default"] / m.r["default"] * f

    rows = []
    for t in tasks_of(a.tasks, a.limit):
        prompt = tok.apply_chat_template(t["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        set_scale(1.0)
        with torch.no_grad():
            g = model.generate(input_ids=torch.tensor([p_ids], device="cuda"), max_new_tokens=900, do_sample=False)
        o_ids = g[0, len(p_ids):].tolist()
        full = p_ids + o_ids
        r = {**t, "prompt_ids": p_ids, "out_ids": o_ids, "text": tok.decode(o_ids), "lora1": score(full, len(p_ids))}
        set_scale(0.5)
        r["lora05"] = score(full, len(p_ids))
        set_scale(1.0)
        with model.disable_adapter():
            r["base"] = score(full, len(p_ids))
        print(f"hf {t['id']}: prompt {len(p_ids)} tokens, output {len(o_ids)} tokens: {r['text'][:160]!r}", flush=True)
        rows.append(r)
    json.dump(rows, open(a.out, "w"))


def run_vllm(a: argparse.Namespace) -> None:
    import numpy as np
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import StructuredOutputsParams

    rows = json.load(open(a.hf))
    schemas = json.load(open(a.schemas))
    kw: Dict[str, Any] = dict(model=a.model, enable_lora=True, max_lora_rank=16, max_loras=1, max_model_len=8192,
                              gpu_memory_utilization=0.85, enforce_eager=True, enable_prefix_caching=False)
    try:
        from vllm.config import StructuredOutputsConfig
        kw["structured_outputs_config"] = StructuredOutputsConfig(backend="xgrammar", disable_any_whitespace=True)
    except Exception as e:  # noqa: BLE001
        print("structured_outputs_config not set:", e)
    llm = LLM(**kw)
    lora = LoRARequest("adapter", 1, a.adapter)
    tok = llm.get_tokenizer()

    def first_diff(x: List[int], y: List[int]) -> int:
        for i, (p, q) in enumerate(zip(x, y)):
            if p != q:
                return i
        return -1 if len(x) == len(y) else min(len(x), len(y))

    def scored(r: Dict[str, Any], use_lora: bool) -> List[float]:
        ids = r["prompt_ids"] + r["out_ids"]
        o = llm.generate([{"prompt_token_ids": ids}], SamplingParams(max_tokens=1, prompt_logprobs=1),
                         lora_request=lora if use_lora else None, use_tqdm=False)[0]
        n = len(r["prompt_ids"])
        return [o.prompt_logprobs[i][ids[i]].logprob for i in range(n, len(ids))]

    summary = []
    for r in rows:
        free = llm.generate([{"prompt_token_ids": r["prompt_ids"]}], SamplingParams(temperature=0, max_tokens=900),
                            lora_request=lora, use_tqdm=False)[0].outputs[0]
        gram = llm.generate([{"prompt_token_ids": r["prompt_ids"]}],
                            SamplingParams(temperature=0, max_tokens=900,
                                           structured_outputs=StructuredOutputsParams(json=schemas[r["act"]])),
                            lora_request=lora, use_tqdm=False)[0].outputs[0]
        lp_l, lp_b = np.array(scored(r, True)), np.array(scored(r, False))
        hf1, hf05, hfb = (np.array(r[k]["lp"]) for k in ("lora1", "lora05", "base"))
        chat = llm.chat([r["messages"]], SamplingParams(max_tokens=1), lora_request=lora, use_tqdm=False,
                        chat_template_kwargs={"enable_thinking": False})[0]
        d_free, d_gram = first_diff(r["out_ids"], list(free.token_ids)), first_diff(r["out_ids"], list(gram.token_ids))
        s = {"id": r["id"], "hf_tokens": len(r["out_ids"]),
             "free": [len(free.token_ids), free.finish_reason, d_free],
             "grammar": [len(gram.token_ids), gram.finish_reason, d_gram],
             "lora_vs_hf1": [float(np.abs(lp_l - hf1).mean()), float(np.abs(lp_l - hf1).max())],
             "lora_vs_hf05": [float(np.abs(lp_l - hf05).mean()), float(np.abs(lp_l - hf05).max())],
             "base_vs_hfbase": [float(np.abs(lp_b - hfb).mean()), float(np.abs(lp_b - hfb).max())],
             "chat_prompt_same": list(chat.prompt_token_ids) == r["prompt_ids"]}
        print(f"\n== {r['id']} ({r['act']}): HF greedy {s['hf_tokens']} tokens")
        print(f"  vLLM free    {s['free'][0]} tokens, finish {s['free'][1]}, first token differing from HF at {d_free}")
        print(f"  vLLM grammar {s['grammar'][0]} tokens, finish {s['grammar'][1]}, first differing at {d_gram}")
        print(f"  teacher-forced |logprob diff| mean/max: adapter vs PEFT scale 1 {s['lora_vs_hf1'][0]:.4f}/{s['lora_vs_hf1'][1]:.3f}; "
              f"vs PEFT scale 0.5 {s['lora_vs_hf05'][0]:.4f}/{s['lora_vs_hf05'][1]:.3f}; base vs base {s['base_vs_hfbase'][0]:.4f}/{s['base_vs_hfbase'][1]:.3f}")
        print(f"  server chat rendering = training prompt ids: {s['chat_prompt_same']}")
        for name, d, out in (("free", d_free, free), ("grammar", d_gram, gram)):
            if 0 <= d < len(r["out_ids"]):
                hf_tok = r["out_ids"][d]
                print(f"  {name} diverges at {d}: HF token {tok.decode([hf_tok])!r} (PEFT logprob {hf1[d]:.3f}, vLLM {lp_l[d]:.3f}); "
                      f"vLLM took {tok.decode([out.token_ids[d]]) if d < len(out.token_ids) else '<end>'!r}; "
                      f"context {tok.decode(r['out_ids'][max(0, d - 12):d])!r}")
        print(f"  vLLM grammar text: {tok.decode(gram.token_ids)[:300]!r}", flush=True)
        summary.append(s)
    json.dump(summary, open(f"vllm_{a.tag}.json", "w"), indent=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["hf", "vllm"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--tasks", nargs="+")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--out", default="hf.json")
    ap.add_argument("--hf", default="hf.json")
    ap.add_argument("--schemas", default="schemas.json")
    ap.add_argument("--tag", default="default")
    a = ap.parse_args()
    run_hf(a) if a.mode == "hf" else run_vllm(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
