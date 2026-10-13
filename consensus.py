"""Step 5: consensus answer and explanation, with majority voting on the same traces."""
import argparse
import json
import random
import zlib
from collections import Counter, defaultdict


def pick(scores, tiebreak, seed):
    if not scores:
        return None
    key = lambda l: (round(scores[l], 12), tiebreak.get(l, 0))
    best = max(key(l) for l in scores)
    tied = sorted(l for l in scores if key(l) == best)
    return tied[0] if len(tied) == 1 else random.Random(seed).choice(tied)


def strength(bundle, W):
    ms = bundle["members"]
    return sum(W.get(m, 0.0) for m in ms) / len(ms) if ms else 1.0


def unfold(root, by_id, W):
    nodes, edges = [], []

    def visit(nid, path):
        if nid not in nodes:
            nodes.append(nid)
        n = by_id.get(nid)
        if n is None or nid in path or not n["support"]:
            return
        b = max(n["support"], key=lambda s: (strength(s, W), len(s["attestations"])))
        for m in b["members"]:
            if [nid, m] not in edges:
                edges.append([nid, m])
            visit(m, path | {nid})

    visit(root, frozenset())
    return nodes, edges


def linearize(nodes, edges, by_id):
    preds = defaultdict(set)
    for c, p in edges:
        preds[c].add(p)
    order, seen = [], set()

    def visit(x, path=frozenset()):
        if x in seen or x in path:
            return
        for p in sorted(preds[x], key=nodes.index):
            visit(p, path | {x})
        seen.add(x)
        order.append(x)

    for x in nodes[::-1]:
        visit(x)
    num = {x: i + 1 for i, x in enumerate(order)}
    lines = []
    for x in order:
        uses = sorted(num[p] for p in preds[x])
        cite = f" (from steps {', '.join(map(str, uses))})" if uses else ""
        lines.append(f"Step {num[x]} [{by_id[x]['type']}]: {by_id[x]['text'].strip()}{cite}")
    return "\n".join(lines)


def consensus(r, seed):
    rng_seed = zlib.crc32(f"{r['question_id']}:{seed}".encode())
    by_id = {n["id"]: n for n in r["ensemble"]["nodes"]}
    W = r["W"]
    label_of = {a["id"]: a["label"] for a in r["answer_clusters"]}
    n_votes = {a["label"]: len(a["traces"]) for a in r["answer_clusters"]}
    votes = Counter(t["pred"] for t in r["traces"])
    rc = pick({label_of[a]: W.get(a, 0.0) for a in label_of}, n_votes, rng_seed)
    mv = pick(dict(votes), {}, rng_seed)
    explanation = ""
    if rc is not None:
        root = next(a for a, l in label_of.items() if l == rc)
        explanation = linearize(*unfold(root, by_id, W), by_id)
    return {"question_id": r["question_id"], "dataset": r["dataset"], "gold": r["gold"],
            "rc": rc, "mv": mv, "votes": dict(votes),
            "scores": {label_of[a]: W.get(a, 0.0) for a in label_of}, "explanation": explanation}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    out = []
    with open(args.scored) as f:
        for line in f:
            out.append(consensus(json.loads(line), args.seed))
    with open(args.out, "w") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    by_ds = defaultdict(list)
    for o in out:
        by_ds[o["dataset"]].append(o)
    for ds, rs in sorted(by_ds.items()):
        acc = lambda m: 100 * sum(o[m] == o["gold"] for o in rs) / len(rs)
        print(f"{ds:12s} n={len(rs):4d}  RC={acc('rc'):.1f}  MV={acc('mv'):.1f}")


if __name__ == "__main__":
    main()
