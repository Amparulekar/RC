"""Step 1: sample reasoning traces from each generator."""
import argparse
import gc
import json
import re

import torch
from vllm import LLM, SamplingParams

GENERATORS = {
    "gemma4-e4b": "google/gemma-4-E4B-it",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "olmo3-7b": "allenai/Olmo-3-7B-Instruct",
    "phi4": "microsoft/phi-4",
}
TEMPERATURE, TOP_P = 0.6, 0.95
MAX_TOKENS = {"gpqa": 8192}
DEFAULT_MAX_TOKENS = 4096
MAX_MODEL_LEN = 16384
LABELS = {"sara": ["ENTAILED", "CONTRADICTED"], "folio": ["UNCERTAIN", "TRUE", "FALSE"]}

SYSTEM = (
    "You are a careful reasoner. For every question, you must:\n"
    "1. Think step by step inside a single <reasoning>...</reasoning> block.\n"
    "2. Put ONLY the final answer (no explanation) inside a <answer>...</answer> tag.\n"
    "Do not write anything outside these two tags. Do not abbreviate your reasoning."
)


def sara_prompt(q):
    return ("You are reasoning about US federal tax law.\n\n"
            f"STATUTE:\n{q['statute']}\n\nCASE SCENARIO:\n{q['case']}\n\nHYPOTHESIS:\n{q['hypothesis']}\n\n"
            "TASK: Decide whether the statute, applied to the case scenario, ENTAILS or "
            "CONTRADICTS the hypothesis. Walk through:\n"
            "  (a) which statutory provisions apply to the case,\n"
            "  (b) what facts in the scenario trigger or fail to trigger those provisions,\n"
            "  (c) what the provisions imply about the hypothesis.\n\n"
            "Output exactly:\n<reasoning>your step-by-step legal analysis</reasoning>\n"
            "<answer>LABEL</answer>   where LABEL is exactly one of ENTAILED or CONTRADICTED")


def gpqa_prompt(q):
    block = "\n".join(f"  {L}. {t}" for L, t in zip("ABCD", q["options"]))
    return ("You are answering a graduate-level science question. Exactly one option is correct.\n\n"
            f"QUESTION:\n{q['question']}\n\nOPTIONS:\n{block}\n\n"
            "TASK: Reason carefully through the underlying science. Walk through:\n"
            "  (a) what principles or equations the question is testing,\n"
            "  (b) how to apply them to the specifics of this problem,\n"
            "  (c) why one option follows and the others do not.\n\n"
            "Output exactly:\n<reasoning>your step-by-step scientific reasoning</reasoning>\n"
            "<answer>LETTER</answer>   where LETTER is exactly one of A, B, C or D (a single capital letter only)")


def musr_prompt(q):
    block = "\n".join(f"  {L}. {t}" for L, t in zip("ABCDEFG", q["options"]))
    return ("You are reasoning about a narrative scenario. Read the narrative carefully, "
            "then choose the single correct answer.\n\n"
            f"NARRATIVE:\n{q['narrative']}\n\nQUESTION:\n{q['question']}\n\nOPTIONS:\n{block}\n\n"
            "TASK: Reason carefully through the evidence in the narrative. Walk through:\n"
            "  (a) what facts the narrative establishes,\n"
            "  (b) how those facts bear on each candidate option,\n"
            "  (c) why one option is best supported and the others are not.\n\n"
            "Output exactly:\n<reasoning>your step-by-step reasoning over the narrative</reasoning>\n"
            f"<answer>LETTER</answer>   where LETTER is exactly one of {', '.join('ABCDEFG'[:len(q['options'])])} "
            "(a single capital letter only)")


def folio_prompt(q):
    return ("You are reasoning about first-order logical entailment in natural language.\n\n"
            f"PREMISES:\n{q['premises']}\n\nCONCLUSION:\n{q['conclusion']}\n\n"
            "TASK: Decide whether the conclusion is logically entailed by the premises. Use these labels:\n"
            "  TRUE      -- the conclusion follows necessarily from the premises\n"
            "  FALSE     -- the negation of the conclusion follows from the premises\n"
            "  UNCERTAIN -- neither the conclusion nor its negation follows\n\n"
            "Walk through:\n  (a) what each premise asserts,\n"
            "  (b) what chain of inferences (if any) the conclusion would require,\n"
            "  (c) whether the premises are sufficient to establish or refute it.\n\n"
            "Output exactly:\n<reasoning>your step-by-step logical analysis</reasoning>\n"
            "<answer>LABEL</answer>   where LABEL is exactly one of TRUE, FALSE or UNCERTAIN")


def medbullets_prompt(q):
    block = "\n".join(f"  {L}. {t}" for L, t in zip("ABCD", q["options"]))
    return ("You are answering a clinical medicine question (USMLE Step 2/3 style). Exactly one option is correct.\n\n"
            f"QUESTION:\n{q['question']}\n\nOPTIONS:\n{block}\n\n"
            "TASK: Reason carefully through the clinical case. Walk through:\n"
            "  (a) the key findings in the case (history, examination, investigations),\n"
            "  (b) what diagnosis or mechanism they point to and how that bears on what is asked,\n"
            "  (c) why one option follows and the others do not.\n\n"
            "Output exactly:\n<reasoning>your step-by-step clinical reasoning</reasoning>\n"
            "<answer>LETTER</answer>   where LETTER is exactly one of A, B, C or D (a single capital letter only)")


PROMPTS = {"sara": sara_prompt, "gpqa": gpqa_prompt, "musr_mm": musr_prompt, "musr_op": musr_prompt,
           "musr_ta": musr_prompt, "folio": folio_prompt, "medbullets": medbullets_prompt}

_REASONING = re.compile(r"<reasoning>(.*?)</reasoning>", re.S | re.I)
_ANSWER = re.compile(r"<answer>((?:(?!<answer>).)*?)</answer>", re.S | re.I)
_FALLBACKS = [
    re.compile(r"<answer>\s*([^<\n]{1,40}?)\s*$", re.I),
    re.compile(r"</answer>\s*([^<\n]{1,40}?)\s*$", re.I),
    re.compile(r"\banswer(?:\s+is)?\s*[:\-]?\s*([^<\n]{1,40}?)\s*\.?\s*</answer>", re.I),
    re.compile(r"(?:^|\n)[ \t*#]*(?:final\s+)?answer[ \t*]*[:\-]?[ \t*]*\n?[ \t*$]*(?:\\boxed\{)?"
               r"([^<\n{}$]{1,40}?)\}?[ \t*.$]*$", re.I),
]


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def match_answer(a, q):
    a = a.strip().strip("*").strip()
    ds = q["dataset"]
    if ds in LABELS:
        u = re.sub(r"[^A-Z]", "", a.upper())
        hits = [lab for lab in LABELS[ds] if u.startswith(lab[:4])]
        if not hits and len(u) >= 5:
            hits = [lab for lab in LABELS[ds] if edit_distance(u, lab) <= 2]
        return hits[0] if len(hits) == 1 else None
    letters = "ABCDEFG"[:len(q["options"])]
    m = re.fullmatch(r"(?:option\s*)?\(?([A-G])\)?[.):]?(\s.*)?", a, re.S | re.I)
    if m and m.group(1).upper() in letters and (m.group(1).isupper() or a.lower().startswith("option")):
        return m.group(1).upper()
    low = a.lower()
    hits = [letters[i] for i, o in enumerate(q["options"]) if o.strip().lower() and o.strip().lower() in low]
    return hits[0] if len(hits) == 1 else None


def parse_output(raw, q):
    rm = _REASONING.search(raw)
    reasoning = rm.group(1).strip() if rm else None
    answer = None
    for a in reversed(_ANSWER.findall(raw)):
        if answer := match_answer(a, q):
            break
    if answer is None:
        for pat in _FALLBACKS:
            for m in reversed(list(pat.finditer(raw))):
                if answer := match_answer(m.group(1), q):
                    break
            if answer:
                break
    return reasoning, answer


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", default=list(GENERATORS), choices=list(GENERATORS))
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tp", type=int, default=1)
    args = ap.parse_args()

    qs = read_jsonl(args.questions)
    with open(args.out, "w") as f:
        for name in args.models:
            llm = LLM(model=GENERATORS[name], tensor_parallel_size=args.tp, max_model_len=MAX_MODEL_LEN,
                      gpu_memory_utilization=0.90, limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0})
            convs, meta, sps = [], [], []
            for q in qs:
                msgs = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": PROMPTS[q["dataset"]](q)}]
                for k in range(args.samples):
                    convs.append(msgs)
                    meta.append((q, k))
                    sps.append(SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, seed=args.seed + k,
                                              max_tokens=MAX_TOKENS.get(q["dataset"], DEFAULT_MAX_TOKENS)))
            outs = llm.chat(convs, sps, chat_template_kwargs={"enable_thinking": False})
            for (q, k), o in zip(meta, outs):
                raw = o.outputs[0].text
                reasoning, answer = parse_output(raw, q)
                f.write(json.dumps({"question_id": q["question_id"], "dataset": q["dataset"], "model": name,
                                    "sample": k, "trace": f"{name}::{k}", "gold": q["gold"], "pred": answer,
                                    "reasoning": reasoning, "raw": raw,
                                    "n_tokens": len(o.outputs[0].token_ids)}, ensure_ascii=False) + "\n")
            del llm
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
