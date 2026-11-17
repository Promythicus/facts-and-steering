#!/usr/bin/env python3
"""Write an invented firm's facts into the quantized 27B file, with forward passes and one closed-form solve.
    python facts.py prompts                      -> out/probe/in.jsonl, out/probe/index.json   (then run probe.sh)
    python facts.py solve <layer> [alpha] [lam] [w]   -> out/facts-L<layer>.patch.pt  (+ fit numbers in out/solve.jsonl)
    python facts.py eval <tag> [inprompt]        asks the model on :8082 every question set -> out/eval-<tag>.jsonl, out/scores.jsonl
The idea (after HomoHabilis exp. 9, v3): at the positions where the answer is written, the model run WITH the
fact in its prompt differs from the model run without it by some change d in the stream after layer L. The
layer's MLP neurons have activations k at that position in the run without the fact. Solve for the smallest
change dW to the MLP's output matrix with dW k = alpha d on those positions and dW k0 = 0 on positions of
questions that must not change (ridge least squares, closed form). No gradients, no training loop.
Environment: FACTS_PROBE (probe folder under out/), FACTS_LAYERS (the layers that probe holds), FACTS_PATCH (patch
file name), FACTS_AUG=1 (round 3: every edit and protected question also with three lead-ins in front, so the edit is
fitted on 4 times as many wordings; the held-out phrasings are untouched), FACTS_THINK=1 (score with thinking on),
FACTS_SYSTEM (score under another system prompt than the one the edit was made under)."""
import json, os, sys, time, urllib.request
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(H, "out"); PR = os.path.join(OUT, os.environ.get("FACTS_PROBE", "probe"))
D = json.load(open(os.path.join(OUT, "data.json"))); SYSTEM = "Answer in a few words."
LAYERS = tuple(int(x) for x in os.environ.get("FACTS_LAYERS", "36,44,50,54,58,61").split(","))
AUG = ("", "Quick question. ", "Hello, I am new here. ", "One thing I need for a file: ") if os.environ.get("FACTS_AUG") else ("",)
THINK = bool(os.environ.get("FACTS_THINK")); EVAL_SYSTEM = os.environ.get("FACTS_SYSTEM", SYSTEM)
LLM = os.environ.get("BENCH_LLM", "http://127.0.0.1:8082/v1/chat/completions")


def chat(q, prefix="", system=SYSTEM):
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{prefix}"


def prompts():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(H, "..", "models", "Qwen3.5-0.8B"))       # same vocabulary as the 27B (248,320 tokens)
    os.makedirs(PR, exist_ok=True); rows, idx = [], []
    for f in D["facts"]:
        ids = tok(f["answer"], add_special_tokens=False).input_ids
        for q in (a + q for q in f["train"] for a in AUG):
            for j in range(len(ids)):
                pre = tok.decode(ids[:j])
                rows.append(chat(q, pre)); idx.append({"kind": "closed", "fact": f["id"], "j": j})
                rows.append(chat(f["statement"] + "\n\n" + q, pre)); idx.append({"kind": "open", "fact": f["id"], "j": j})
        for q in (a + q for q in f["others"] for a in AUG):
            rows.append(chat(q)); idx.append({"kind": "protect", "fact": f["id"]})
    for g in D["general"]:
        for a in AUG:
            if g["use"] == "protect":
                rows.append(chat(a + g["q"])); idx.append({"kind": "protect", "fact": -1})
    with open(os.path.join(PR, "in.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps({"text": r}) + "\n")
    json.dump(idx, open(os.path.join(PR, "index.json"), "w")); print(len(rows), "texts;", sum(i["kind"] == "closed" for i in idx), "edit positions;", sum(i["kind"] == "protect" for i in idx), "protected")


def load(name):
    m = json.load(open(os.path.join(PR, "meta.json"))); L, d = m["shape"][name]
    return np.fromfile(os.path.join(PR, name + ".f32"), dtype=np.float32).reshape(-1, L, d)


def solve(layer, alpha=1.0, lam=0.01, w=4.0):
    import torch
    idx = json.load(open(os.path.join(PR, "index.json"))); li = sorted(LAYERS).index(layer)
    K_all, R_all = load("ffn_swiglu")[:, li].astype(np.float64), load("l_out")[:, li].astype(np.float64)
    c = [i for i, x in enumerate(idx) if x["kind"] == "closed"]; o = [i + 1 for i in c]; p = [i for i, x in enumerate(idx) if x["kind"] == "protect"]
    K, Dt, K0 = K_all[c], alpha * (R_all[o] - R_all[c]), K_all[p]
    A = np.concatenate([K, np.sqrt(w) * K0]); T = np.concatenate([Dt, np.zeros((len(K0), Dt.shape[1]))])
    G = A @ A.T; dW = (np.linalg.solve(G + lam * np.trace(G) / len(G) * np.eye(len(G)), T)).T @ A          # [hidden, neurons]
    fit = {"layer": layer, "alpha": alpha, "lam": lam, "w": w, "edit_positions": len(K), "protected": len(K0),
           "target_norm": round(float(np.linalg.norm(Dt, axis=1).mean()), 3), "stream_norm": round(float(np.linalg.norm(R_all[c], axis=1).mean()), 3),
           "miss_on_edit": round(float(np.linalg.norm(K @ dW.T - Dt) / np.linalg.norm(Dt)), 4), "leak_on_protected": round(float(np.linalg.norm(K0 @ dW.T, axis=1).mean()), 4),
           "dW_norm": round(float(np.linalg.norm(dW)), 3)}
    torch.save({f"blk.{layer}.ffn_down.weight": {"columns": torch.arange(dW.shape[1]), "added": torch.tensor(dW, dtype=torch.float32)}}, os.path.join(OUT, os.environ.get("FACTS_PATCH", f"facts-L{layer}.patch.pt")))
    open(os.path.join(OUT, "solve.jsonl"), "a").write(json.dumps(fit) + "\n"); print(json.dumps(fit))


def ask(q, system=SYSTEM):
    body = {"model": "local", "temperature": 0, "max_tokens": 1500 if THINK else 40, "chat_template_kwargs": {"enable_thinking": THINK}, "messages": [{"role": "system", "content": system}, {"role": "user", "content": q}]}
    r = json.load(urllib.request.urlopen(urllib.request.Request(LLM, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=300))
    return (r["choices"][0]["message"].get("content") or "").strip()


def has(answer, target):
    n = lambda s: "".join(ch for ch in s.lower() if ch.isalnum() or ch == " ")
    return n(target) in n(answer)


def evaluate(tag, inprompt=False):
    system = EVAL_SYSTEM + ("\n\nFacts about the law firm " + D["firm"] + ":\n" + "\n".join(f["statement"] for f in D["facts"]) if inprompt else "")
    rows = []
    for f in D["facts"]:
        for kind in (("held", "others") if THINK else ("train", "held", "others")):
            for q in f[kind]:
                a = ask(q, system); rows.append({"set": kind, "fact": f["id"], "q": q, "target": f["answer"], "answer": a, "hit": has(a, f["answer"])})
    for g in D["general"]:
        a = ask(g["q"], system); rows.append({"set": "general-" + g["use"], "fact": -1, "q": g["q"], "target": g["a"], "answer": a, "hit": has(a, g["a"])})
    with open(os.path.join(OUT, f"eval-{tag}.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    share = lambda s: round(sum(r["hit"] for r in rows if r["set"] == s) / max(1, sum(r["set"] == s for r in rows)), 4)
    learned = sum(sum(r["hit"] for r in rows if r["set"] == "held" and r["fact"] == f["id"]) >= 3 for f in D["facts"])
    s = {"tag": tag, "held_out": share("held"), "train": share("train"), "facts_learned": f"{learned}/{len(D['facts'])}", "bleed_to_other_firms": share("others"),
         "general_score": share("general-score"), "general_protect": share("general-protect")}
    open(os.path.join(OUT, "scores.jsonl"), "a").write(json.dumps(s) + "\n"); print(json.dumps(s))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "prompts":
        prompts()
    elif a[0] == "solve":
        solve(int(a[1]), *map(float, a[2:]))
    else:
        evaluate(a[1], len(a) > 2)
