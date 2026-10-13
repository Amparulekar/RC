"""Step 4: score every node of the ensemble DAG."""
import argparse
import json
from collections import defaultdict


def topo_order(nodes):
    by_id = {n["id"]: n for n in nodes}
    children, indeg = defaultdict(list), {n["id"]: 0 for n in nodes}
    for n in nodes:
        for p in {m for k in ("support", "defeats") for b in n[k] for m in b["members"] if m in by_id}:
            children[p].append(n["id"])
            indeg[n["id"]] += 1
    order = [x for x in indeg if indeg[x] == 0]
    for u in order:
        for c in children[u]:
            indeg[c] -= 1
            if indeg[c] == 0:
                order.append(c)
    return order


def strength(bundle, W):
    ms = bundle["members"]
    return sum(W.get(m, 0.0) for m in ms) / len(ms) if ms else 1.0


def score(nodes, traces):
    by_id = {n["id"]: n for n in nodes}
    attest = lambda ts: len(set(ts)) / (len(traces) or 1)
    W, fallback_answers = {}, []
    for nid in topo_order(nodes):
        n = by_id[nid]
        sup = n["support"]
        sup_tr = {t for b in sup for t in b["attestations"]}
        def_tr = {t for b in n["defeats"] for t in b["attestations"]}
        if n["type"] == "Answer" and not sup:
            fallback_answers.append(nid)
            continue
        if not sup or (n["type"] == "Fact" and set(n["source_traces"]) - sup_tr):
            a_plus, p_plus = attest(n["source_traces"]), 1.0
        else:
            a_plus = attest(n["source_traces"] if n["type"] == "Answer" else sup_tr)
            p_plus = max(strength(b, W) for b in sup)
        p_minus = max((strength(b, W) for b in n["defeats"]), default=0.0)
        W[nid] = max(0.0, a_plus * p_plus - attest(def_tr) * p_minus)
    structured = [W[x] / (attest(by_id[x]["source_traces"]) or 1) for x in W if by_id[x]["type"] == "Answer"]
    imputed = sum(structured) / len(structured) if structured else 1.0
    for nid in fallback_answers:
        W[nid] = attest(by_id[nid]["source_traces"]) * imputed
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.merged) as f, open(args.out, "w") as g:
        for line in f:
            r = json.loads(line)
            W = score(r["ensemble"]["nodes"], [t["trace"] for t in r["traces"]])
            g.write(json.dumps({**r, "W": {k: round(v, 8) for k, v in W.items()}}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
