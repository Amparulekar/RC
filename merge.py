"""Step 3: merge the per-trace DAGs of each question into one ensemble DAG."""
import argparse
import gc
import json
import re
from collections import defaultdict

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
JUDGE = "Qwen/Qwen3.8-27B"
MERGE_TYPES = ["Planning", "Fact", "Reasoning", "Conclusion"]
MAX_CLUSTER = 12
MAX_MODEL_LEN = 16384

VERIFY_SYSTEM = ("You are a strict judge of semantic equivalence between statements extracted from reasoning "
                 "traces. You group statements that make the SAME claim. Output only JSON.")
PAIR_SYSTEM = ("You are a strict semantic equivalence judge for reasoning-graph nodes. Two nodes should merge "
               "ONLY IF they express the SAME underlying claim, fact, or inference, regardless of wording. "
               "Lexical overlap is NOT enough. Answer with a single token: YES or NO.")
GROUPS_SCHEMA = {"type": "object", "properties": {"groups": {"type": "array", "items": {
    "type": "array", "items": {"type": "integer"}}}}, "required": ["groups"], "additionalProperties": False}


def question_context(q):
    ds = q["dataset"]
    if ds == "sara":
        return f"STATUTE:\n{q['statute']}\n\nCASE:\n{q['case']}\n\nHYPOTHESIS:\n{q['hypothesis']}"
    if ds == "folio":
        return f"PREMISES:\n{q['premises']}\n\nCONCLUSION:\n{q['conclusion']}"
    opts = "\n".join(f"{L}. {t}" for L, t in zip("ABCDEFG", q["options"]))
    head = f"NARRATIVE:\n{q['narrative']}\n\n" if ds.startswith("musr") else ""
    return f"{head}QUESTION:\n{q['question']}\n\nOPTIONS:\n{opts}"


def verify_prompt(q, ntype, texts):
    listing = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    return (f"QUESTION CONTEXT:\n{question_context(q)}\n\n----------------------------------------\n"
            f"The numbered statements below are \"{ntype}\" steps taken from different reasoning traces about "
            "this question. Group together the statements that state the SAME claim (same entities, same "
            "assertion, same polarity), regardless of wording.\n"
            "Rules:\n"
            "  - Lexical overlap is not enough. Different entities, numbers, provisions, options or inferences "
            "are DIFFERENT claims.\n"
            "  - A general rule and its application to a specific case are DIFFERENT claims.\n"
            "  - A claim and its negation are DIFFERENT claims.\n"
            "  - Every statement appears in exactly one group; a statement that matches no other is a group of "
            "its own.\n\n"
            f"STATEMENTS:\n{listing}\n\n"
            'Output JSON: {"groups": [[1, 3], [2], ...]}')


def pair_prompt(q, ntype, a, b):
    return (f"QUESTION CONTEXT:\n{question_context(q)}\n\n----------------------------------------\n"
            f"Both nodes are of type \"{ntype}\".\nNODE A: {a}\nNODE B: {b}\n\n"
            "Should A and B be merged because they assert the SAME underlying claim / fact / inference?\n"
            "Reply with exactly one token: YES or NO.")


def norm_label(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", (s or "").strip().lower()))


def agglomerate(emb, tau):
    if len(emb) == 1:
        return [[0]]
    lab = fcluster(linkage(np.clip(pdist(emb, "cosine"), 0, 2), "average"), t=1 - tau, criterion="distance")
    out = defaultdict(list)
    for i, l in enumerate(lab):
        out[l].append(i)
    return list(out.values())


def split_same_trace(idx, traces, sim, cap=MAX_CLUSTER):
    if len(idx) <= cap and len({traces[i] for i in idx}) == len(idx):
        return [idx]
    centr = sim[np.ix_(idx, idx)].mean(1)
    subs = []
    for i in [idx[j] for j in np.argsort(-centr)]:
        ok = [s for s in subs if len(s) < cap and traces[i] not in {traces[k] for k in s}]
        if ok:
            max(ok, key=lambda s: sim[i, s].mean()).append(i)
        else:
            subs.append([i])
    return subs


def parse_groups(text, n):
    try:
        groups = json.loads(text[text.find("{"):text.rfind("}") + 1]).get("groups", [])
    except (ValueError, AttributeError):
        return [[i] for i in range(n)]
    seen, out = set(), []
    for g in groups if isinstance(groups, list) else []:
        g = [i - 1 for i in (g if isinstance(g, list) else []) if isinstance(i, int) and 1 <= i <= n]
        g = [i for i in dict.fromkeys(g) if i not in seen]
        seen.update(g)
        if g:
            out.append(g)
    return out + [[i] for i in range(n) if i not in seen]


class UnionFind:
    def __init__(self, items):
        self.p = {x: x for x in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def globalize(rec):
    t = rec["trace"]
    g = lambda x: f"{t}::{x}"
    return [{"id": g(n["id"]), "type": n["type"], "text": n["text"], "trace": t,
             "support": [[g(x) for x in b] for b in n["support"]],
             "defeats": [[g(x) for x in b] for b in n["defeats"]]} for n in rec["dag"]["nodes"]]


def merge_dags(gnodes, cluster_of):
    cl = {}
    for n in gnodes:
        c = cluster_of[n["id"]]
        e = cl.setdefault(c, {"id": c, "type": n["type"], "text": n["text"], "members": [],
                              "source_traces": set(), "support": {}, "defeats": {}})
        e["members"].append({"trace": n["trace"], "text": n["text"]})
        e["source_traces"].add(n["trace"])
    for n in gnodes:
        e = cl[cluster_of[n["id"]]]
        for k in ("support", "defeats"):
            for b in n[k]:
                key = frozenset(cluster_of[x] for x in b)
                if e["id"] not in key:
                    e[k].setdefault(key, set()).add(n["trace"])
    fin = lambda d: sorted(({"members": sorted(k), "attestations": sorted(v)} for k, v in d.items()),
                           key=lambda b: (-len(b["attestations"]), len(b["members"]), b["members"]))
    return [{**e, "source_traces": sorted(e["source_traces"]), "n_attest": len(e["source_traces"]),
             "support": fin(e["support"]), "defeats": fin(e["defeats"])} for e in cl.values()]


def find_cycle(by_id):
    def edges(u):
        for kind in ("support", "defeats"):
            for bi, b in enumerate(by_id[u][kind]):
                for v in b["members"]:
                    yield v, (u, kind, bi)
    color, via = {}, {}
    for s in by_id:
        if color.get(s):
            continue
        color[s] = 1
        stack = [(s, edges(s))]
        while stack:
            u, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                color[u] = 2
                stack.pop()
                continue
            v, lab = nxt
            if color.get(v) == 1:
                cyc, w = [lab], u
                while w != v:
                    cyc.append(via[w][1])
                    w = via[w][0]
                return cyc
            if not color.get(v):
                color[v], via[v] = 1, (u, lab)
                stack.append((v, edges(v)))
    return None


def break_cycles(nodes):
    by_id = {n["id"]: n for n in nodes}
    while cyc := find_cycle(by_id):
        nid, kind, bi = min(cyc, key=lambda e: (len(by_id[e[0]][e[1]][e[2]]["attestations"]),
                                                -len(by_id[e[0]][e[1]][e[2]]["members"])))
        del by_id[nid][kind][bi]
    for n in nodes:
        n["gate"] = "Atomic" if not n["support"] else "And" if len(n["support"]) == 1 else "Or"


def plan_question(q, recs, embed, tau, method):
    gnodes = [n for r in recs for n in globalize(r)]
    pred_of = {r["trace"]: r["pred"] for r in recs}
    st = {"q": q, "traces": recs, "gnodes": gnodes, "requests": [], "groups": [],
          "uf": UnionFind([n["id"] for n in gnodes])}
    by_label = defaultdict(list)
    for n in gnodes:
        if n["type"] == "Answer":
            by_label[norm_label(n["text"])].append(n["id"])
    for ids in by_label.values():
        for x in ids[1:]:
            st["uf"].union(ids[0], x)
    parts = defaultdict(list)
    for n in gnodes:
        if n["type"] in MERGE_TYPES:
            parts[(n["type"], norm_label(pred_of[n["trace"]]) if n["type"] == "Conclusion" else "")].append(n)
    for (ntype, _), ns in parts.items():
        if len(ns) < 2:
            continue
        emb = embed([n["text"] for n in ns])
        sim = emb @ emb.T
        tr = [n["trace"] for n in ns]
        if method == "pairwise":
            for i in range(len(ns)):
                for j in range(i + 1, len(ns)):
                    if tr[i] != tr[j] and sim[i, j] >= tau:
                        st["requests"].append(("pair", ntype, [ns[i]["id"], ns[j]["id"]],
                                               [ns[i]["text"], ns[j]["text"]]))
            continue
        for c in agglomerate(emb, tau):
            for s in split_same_trace(c, tr, sim):
                if len(s) < 2:
                    continue
                ids, texts = [ns[i]["id"] for i in s], [ns[i]["text"] for i in s]
                if method == "embed":
                    st["groups"].append(ids)
                else:
                    st["requests"].append(("group", ntype, ids, texts))
    return st


def finish_question(st, answers):
    uf, groups = st["uf"], list(st["groups"])
    for (kind, _, ids, _), a in zip(st["requests"], answers):
        if kind == "pair":
            if a.strip().upper().startswith("YES"):
                uf.union(*ids)
        else:
            groups += [[ids[i] for i in g] for g in parse_groups(a, len(ids))]
    for g in groups:
        for x in g[1:]:
            uf.union(g[0], x)
    cluster_of = {n["id"]: uf.find(n["id"]) for n in st["gnodes"]}
    nodes = merge_dags(st["gnodes"], cluster_of)
    break_cycles(nodes)
    answers_cl = defaultdict(set)
    for n in st["gnodes"]:
        if n["type"] == "Answer":
            answers_cl[cluster_of[n["id"]]].add(n["trace"])
    label = {n["id"]: n["text"] for n in nodes}
    q = st["q"]
    return {"question_id": q["question_id"], "dataset": q["dataset"], "gold": q["gold"],
            "traces": [{"trace": r["trace"], "model": r["model"], "pred": r["pred"], "fallback": r["fallback"]}
                       for r in st["traces"]],
            "ensemble": {"nodes": nodes},
            "answer_clusters": sorted(({"id": c, "label": label[c], "traces": sorted(t)}
                                       for c, t in answers_cl.items()), key=lambda a: -len(a["traces"])),
            "n_llm_calls": len(st["requests"])}


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--dags", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=0.55)
    ap.add_argument("--merge", choices=["cluster_verify", "pairwise", "embed"], default="cluster_verify")
    ap.add_argument("--tp", type=int, default=2)
    args = ap.parse_args()

    qs = {q["question_id"]: q for q in read_jsonl(args.questions)}
    by_q = defaultdict(list)
    for r in read_jsonl(args.dags):
        by_q[r["question_id"]].append(r)

    embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
    embed = lambda texts: embedder.encode(texts, batch_size=128, convert_to_numpy=True,
                                          normalize_embeddings=True, show_progress_bar=False)
    states = [plan_question(qs[qid], recs, embed, args.tau, args.merge) for qid, recs in by_q.items() if qid in qs]

    convs, sps, owner = [], [], []
    if any(st["requests"] for st in states):
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        llm = LLM(model=JUDGE, tensor_parallel_size=args.tp, max_model_len=MAX_MODEL_LEN,
                  gpu_memory_utilization=0.90, enable_prefix_caching=True,
                  limit_mm_per_prompt={"image": 0, "video": 0})
        sp_group = SamplingParams(temperature=0.0, max_tokens=1024, seed=0,
                                  structured_outputs=StructuredOutputsParams(json=GROUPS_SCHEMA))
        sp_pair = SamplingParams(temperature=0.0, max_tokens=2, seed=0,
                                 structured_outputs=StructuredOutputsParams(choice=["YES", "NO"]))
        for si, st in enumerate(states):
            for kind, ntype, _, texts in st["requests"]:
                if kind == "pair":
                    convs.append([{"role": "system", "content": PAIR_SYSTEM},
                                  {"role": "user", "content": pair_prompt(st["q"], ntype, *texts)}])
                else:
                    convs.append([{"role": "system", "content": VERIFY_SYSTEM},
                                  {"role": "user", "content": verify_prompt(st["q"], ntype, texts)}])
                sps.append(sp_pair if kind == "pair" else sp_group)
                owner.append(si)
        outs = llm.chat(convs, sps, chat_template_kwargs={"enable_thinking": False})
        del llm
        gc.collect()
        torch.cuda.empty_cache()
    else:
        outs = []

    answers = [[] for _ in states]
    for si, o in zip(owner, outs):
        answers[si].append(o.outputs[0].text)
    with open(args.out, "w") as f:
        for st, a in zip(states, answers):
            f.write(json.dumps(finish_question(st, a), ensure_ascii=False) + "\n")
    print(f"{len(states)} questions merged, {len(convs)} verifier calls")


if __name__ == "__main__":
    main()
