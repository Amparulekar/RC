"""Step 2: extract a typed reasoning DAG from every trace."""
import argparse
import gc
import json
import re
from collections import defaultdict

import torch
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

EXTRACTOR = "Qwen/Qwen3.8-27B"
TEMPERATURE, TOP_P, SEED = 0.3, 0.95, 0
MAX_TOKENS = 4096
MAX_MODEL_LEN = 20480
MAX_TRACE_CHARS = 40000
LABELS = {"sara": ["ENTAILED", "CONTRADICTED"], "folio": ["UNCERTAIN", "TRUE", "FALSE"]}
TYPES = ["Planning", "Fact", "Reasoning", "Conclusion"]

SYSTEM = ("You are a structured-reasoning extractor. Given a question, a reasoning trace and its final answer, "
          "you produce a directed acyclic graph (DAG) in JSON that captures the trace's logical structure. "
          "Output only the JSON object.")

INSTRUCTIONS = """TASK: Extract a Type-B reasoning DAG from the reasoning trace below.

NODE TYPES (use exactly these strings):
  - "Planning":   meta-reasoning, problem decomposition, deciding what to check
  - "Fact":       a factual statement drawn from the question context (statute, scenario, narrative, premises,
                  options); its text should restate the relevant span of the question
  - "Reasoning":  an intermediate inferential step linking facts to the outcome
  - "Conclusion": the final determination

REQUIRED: exactly ONE node of type "Conclusion", whose text expresses the final answer given below (in the
trace's own words, but the same verdict). All other nodes must transitively support this Conclusion.

Each node has:
  - "id":      unique short id ("p1", "f1", "r1", "c1", ...)
  - "type":    one of the four types above
  - "text":    one self-contained sentence (resolve pronouns and references like "option B" or "the
               hypothesis" using the question context)
  - "support": list of bundles. A bundle is a list of node ids that JOINTLY (AND) justify this node;
               different bundles are alternative, independent (OR) justifications.
  - "defeats": list of bundles whose joint truth would SUPPRESS this node; [] unless the trace itself
               states a counter-consideration. Do not invent defeaters.
  - "rejects": the ANSWER LABELS (listed below the question) that this step explicitly argues AGAINST, e.g.
               "Option A is wrong because ..." -> ["A"]; "the conclusion cannot be shown to be true" -> ["TRUE"].
               [] for most steps. Only what the trace itself says; never the final answer's own label.

RULES:
  - List nodes premises first: all Fact nodes first, then Planning and Reasoning nodes, the Conclusion last.
  - A bundle may only reference ids of nodes listed EARLIER (this keeps the graph acyclic).
  - Facts may have empty support (atomic premises from the question).
  - Planning, Reasoning and Conclusion nodes MUST have at least one support bundle.
  - A bundle names only the few steps that directly justify the node (at most 10 ids); at most 4 bundles per node.
  - Do NOT add information the trace does not contain. One node per distinct claim; at most 60 nodes."""

BUNDLES = {"type": "array", "maxItems": 4,
           "items": {"type": "array", "minItems": 1, "maxItems": 10, "items": {"type": "string", "maxLength": 12}}}


def answer_labels(q):
    return LABELS[q["dataset"]] if q["dataset"] in LABELS else list("ABCDEFG"[:len(q["options"])])


def json_schema(labels):
    return {
        "type": "object",
        "properties": {"nodes": {"type": "array", "minItems": 1, "maxItems": 60, "items": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "maxLength": 12},
                "type": {"enum": TYPES},
                "text": {"type": "string", "maxLength": 400},
                "support": BUNDLES,
                "defeats": BUNDLES,
                "rejects": {"type": "array", "maxItems": 4, "items": {"enum": list(labels)}},
            },
            "required": ["id", "type", "text", "support", "defeats", "rejects"],
            "additionalProperties": False}}},
        "required": ["nodes"],
        "additionalProperties": False,
    }


def question_context(q):
    ds = q["dataset"]
    if ds == "sara":
        return f"STATUTE:\n{q['statute']}\n\nCASE:\n{q['case']}\n\nHYPOTHESIS:\n{q['hypothesis']}"
    if ds == "folio":
        return f"PREMISES:\n{q['premises']}\n\nCONCLUSION:\n{q['conclusion']}"
    opts = "\n".join(f"{L}. {t}" for L, t in zip("ABCDEFG", q["options"]))
    head = f"NARRATIVE:\n{q['narrative']}\n\n" if ds.startswith("musr") else ""
    return f"{head}QUESTION:\n{q['question']}\n\nOPTIONS:\n{opts}"


def user_prompt(q, answer, reasoning):
    return (f"{INSTRUCTIONS}\n\n----------------------------------------\nQUESTION CONTEXT:\n{question_context(q)}\n\n"
            f"ANSWER LABELS: {', '.join(answer_labels(q))}\n\n"
            f"FINAL ANSWER (the Conclusion node's verdict):\n{answer}\n\n"
            f"REASONING TRACE:\n{reasoning[:MAX_TRACE_CHARS]}\n----------------------------------------\n\n"
            "Output only the JSON object {\"nodes\": [...]}.")


def trace_text(r):
    if r.get("reasoning"):
        return r["reasoning"]
    return (r.get("raw") or "").replace("<reasoning>", "").split("<answer>")[0].strip()


# ---------------------------------------------------------------- parsing and repair

_WORD = re.compile(r"[a-z0-9]+")
_LATEX = re.compile(r"(?<!\\)\\(?=(neg|text|textbf|frac|dfrac|to|times|cdot|rightarrow|Rightarrow|leftarrow|"
                    r"leftrightarrow|Leftrightarrow|implies|iff|land|lor|lnot|forall|exists|in|notin|sqrt|approx|"
                    r"leq|geq|le|ge|neq|ne|pm|mp|infty|sum|prod|int|partial|propto|sim|simeq|equiv|subset|subseteq|"
                    r"cup|cap|emptyset|mathrm|mathbf|mathit|mathcal|left|right|hbar|alpha|beta|gamma|delta|Delta|"
                    r"epsilon|varepsilon|theta|lambda|mu|nu|pi|rho|sigma|Sigma|tau|phi|omega|Omega|circ|degree|"
                    r"ldots|cdots|quad|langle|rangle|vee|wedge|bot|top|therefore|because|ce)(?![A-Za-z]))")


def parse_json(raw):
    blob = _LATEX.sub(r"\\\\", raw)
    blob = re.sub(r"[\t\x08\x0c\r](?=[A-Za-z])",
                  lambda c: {"\t": "\\\\t", "\x08": "\\\\b", "\x0c": "\\\\f", "\r": "\\\\r"}[c.group(0)], blob)
    start = blob.find("{")
    if start < 0:
        return None
    blob = blob[start:]
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        pass
    for end in [m.end() for m in re.finditer(r"\}\s*,?", blob)][::-1][:200]:
        try:
            obj = json.loads(blob[:end].rstrip().rstrip(",") + "]}")
            if isinstance(obj, dict) and obj.get("nodes"):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def ids_in(bundles):
    return [x for b in bundles for x in b]


def find_cycle(nodes):
    by_id = {n["id"]: n for n in nodes}
    color, parent = {}, {}

    def preds(nid):
        n = by_id[nid]
        return [x for x in ids_in(n.get("support", [])) + ids_in(n.get("defeats", [])) if x in by_id]

    for s in by_id:
        if color.get(s):
            continue
        stack = [(s, iter(preds(s)))]
        color[s] = 1
        while stack:
            u, it = stack[-1]
            v = next(it, None)
            if v is None:
                color[u] = 2
                stack.pop()
            elif color.get(v) == 1:
                cyc, w = [(u, v)], u
                while w != v:
                    cyc.append((parent[w], w))
                    w = parent[w]
                return cyc
            elif not color.get(v):
                color[v], parent[v] = 1, u
                stack.append((v, iter(preds(v))))
    return None


def is_valid(nodes):
    ids = {n["id"] for n in nodes}
    for n in nodes:
        for k in ("support", "defeats"):
            if any(not b or any(x not in ids for x in b) for b in n[k]):
                return False
        if n["type"] != "Fact" and not n["support"]:
            return False
    return sum(n["type"] == "Conclusion" for n in nodes) == 1 and not find_cycle(nodes)


def content_words(s):
    return [w for w in _WORD.findall(s.lower()) if len(w) > 3]


def restates_question(text, context_words):
    ws = content_words(text)
    if len(ws) < 3:
        return False
    frac = sum(w in context_words for w in ws) / len(ws)
    return frac >= 0.9 and not re.search(r"\b(so|thus|therefore|hence|implies|means|because|since)\b", text, re.I)


def mentions_answer(text, answer, options=None):
    t, a = text.lower(), answer.lower()
    if len(answer) == 1:
        score = 2 * bool(re.search(rf"(option|answer|choice)\s*\(?{a}\b|\({a}\)|\b{a}[.):]", t))
        if options and ord(answer) - 65 < len(options):
            opt = options[ord(answer) - 65].strip().lower()
            score += bool(opt) and opt in t
        return score
    return 2 * bool(re.search(rf"\b{re.escape(a)}", t))


def repair(obj, answer, context="", options=None):
    raw_nodes = obj.get("nodes") if isinstance(obj, dict) else None
    if not isinstance(raw_nodes, list):
        return None
    nodes, seen = [], set()
    type_map = {t.lower(): t for t in TYPES}
    for n in raw_nodes:
        if not isinstance(n, dict) or n.get("id") in (None, "") or str(n["id"]) in seen:
            continue
        nid = str(n["id"])
        seen.add(nid)
        bundles = {}
        for k in ("support", "defeats"):
            v = n.get(k) or []
            v = v if isinstance(v, list) else [v]
            bundles[k] = [[str(x) for x in (b if isinstance(b, list) else [b]) if x not in (None, "")] for b in v]
        rejects = [str(x) for x in (n.get("rejects") or []) if isinstance(x, str) and x != answer]
        nodes.append({"id": nid, "type": type_map.get(str(n.get("type", "")).strip().lower(), "Reasoning"),
                      "text": str(n.get("text") or "").strip(), **bundles, "rejects": list(dict.fromkeys(rejects))})
    nodes = [n for n in nodes if n["text"]] or nodes
    if not nodes:
        return None
    ids = {n["id"] for n in nodes}

    for n in nodes:
        for k in ("support", "defeats"):
            out = []
            for b in n[k]:
                kept = list(dict.fromkeys(x for x in b if x in ids and x != n["id"]))
                if kept and kept not in out:
                    out.append(kept)
            n[k] = out

    concs = [n for n in nodes if n["type"] == "Conclusion"]
    if len(concs) > 1:
        best = max(concs, key=lambda n: (mentions_answer(n["text"], answer, options), nodes.index(n)))
        for n in concs:
            if n is not best:
                n["type"] = "Reasoning"
    elif not concs:
        has_succ = {x for n in nodes for k in ("support", "defeats") for x in ids_in(n[k])}
        sinks = [n["id"] for n in nodes if n["id"] not in has_succ and n["type"] != "Planning"]
        nodes.append({"id": "c_added", "type": "Conclusion", "text": f"The answer is {answer}.",
                      "support": [sinks[-1:] or [nodes[-1]["id"]]], "defeats": [], "rejects": []})

    context_words = set(content_words(context))
    for _ in range(50):
        changed = False
        order = {n["id"]: i for i, n in enumerate(nodes)}
        for i, n in enumerate(nodes):
            if n["type"] == "Fact" or n["support"]:
                continue
            changed = True
            if n["type"] == "Reasoning" and restates_question(n["text"], context_words):
                n["type"] = "Fact"
                continue
            prev = [m for m in nodes[:i][::-1] if m["type"] in ("Planning", "Fact")]
            atomic = [m for m in nodes if m["type"] == "Fact" and not m["support"] and not m["defeats"] and m is not n]
            anchor = (prev or atomic or [None])[0]
            if anchor is not None:
                n["support"] = [[anchor["id"]]]
            elif n["type"] == "Conclusion":
                others = [m["id"] for m in nodes if m is not n]
                if not others:
                    return None
                n["support"] = [[others[-1]]]
            else:
                n["type"] = "Fact"
        cyc = find_cycle(nodes)
        if cyc:
            child, par = max(cyc, key=lambda e: order[e[1]] - order[e[0]])
            c = next(n for n in nodes if n["id"] == child)
            for k in ("support", "defeats"):
                c[k] = [b for b in ([x for x in b if x != par] for b in c[k]) if b]
            changed = True
        if not changed:
            break
    return nodes if is_valid(nodes) else None


def with_gates(nodes):
    for n in nodes:
        n["gate"] = "Atomic" if not n["support"] else "And" if len(n["support"]) == 1 else "Or"
    return nodes


def build_dag(raw, answer, context, options):
    obj = parse_json(raw)
    nodes = repair(obj, answer, context, options) if obj is not None else None
    if nodes is None:
        return with_gates([{"id": "ans", "type": "Answer", "text": answer, "support": [], "defeats": [],
                            "rejects": []}]), True
    conc = next(n["id"] for n in nodes if n["type"] == "Conclusion")
    aid = "ans"
    while any(n["id"] == aid for n in nodes):
        aid += "_"
    nodes.append({"id": aid, "type": "Answer", "text": answer, "support": [[conc]], "defeats": [], "rejects": []})
    return with_gates(nodes), False


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tp", type=int, default=2)
    args = ap.parse_args()

    qs = {q["question_id"]: q for q in read_jsonl(args.questions)}
    traces = [r for r in read_jsonl(args.traces) if r["question_id"] in qs]
    todo = [r for r in traces if r["pred"] is not None]

    llm = LLM(model=EXTRACTOR, tensor_parallel_size=args.tp, max_model_len=MAX_MODEL_LEN,
              gpu_memory_utilization=0.90, enable_prefix_caching=True, limit_mm_per_prompt={"image": 0, "video": 0})
    sps = {}

    def sampling(q):
        key = tuple(answer_labels(q))
        if key not in sps:
            sps[key] = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, seed=SEED, max_tokens=MAX_TOKENS,
                                      structured_outputs=StructuredOutputsParams(json=json_schema(key)))
        return sps[key]

    convs = [[{"role": "system", "content": SYSTEM},
              {"role": "user", "content": user_prompt(qs[r["question_id"]], r["pred"], trace_text(r))}] for r in todo]
    outs = llm.chat(convs, [sampling(qs[r["question_id"]]) for r in todo],
                    chat_template_kwargs={"enable_thinking": False})

    stats = defaultdict(int)
    with open(args.out, "w") as f:
        for r, o in zip(todo, outs):
            q = qs[r["question_id"]]
            nodes, fallback = build_dag(o.outputs[0].text, r["pred"], question_context(q), q.get("options"))
            stats["fallback"] += fallback
            f.write(json.dumps({"question_id": r["question_id"], "dataset": r["dataset"], "model": r["model"],
                                "sample": r["sample"], "trace": r["trace"], "gold": r["gold"], "pred": r["pred"],
                                "fallback": fallback, "dag": {"nodes": nodes},
                                "n_in": len(o.prompt_token_ids), "n_out": len(o.outputs[0].token_ids)},
                               ensure_ascii=False) + "\n")
    print(f"{len(todo)} traces extracted, {stats['fallback']} answer-only fallbacks, "
          f"{len(traces) - len(todo)} without an answer")
    del llm
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
