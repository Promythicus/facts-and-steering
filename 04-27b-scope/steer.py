#!/usr/bin/env python3
"""Steering the quantized 27B at run time: a direction added to the stream while it answers. No weight is changed;
this is the screen for which directions are worth writing into the file later, and the source of the fun examples.
    python steer.py prompts              -> out/steer/in.jsonl       (then the probe, see run.sh)
    python steer.py vectors              -> out/steer/<trait>.gguf    (llama.cpp control vectors, layers 24-31 of 64)
    python steer.py demo <trait> <scale>    asks the server on :8082 eight questions the direction was not made from
                                            -> out/steer/demo.jsonl
A trait's direction at a layer = mean stream at the last prompt token over 32 questions asked under the trait's
system prompt, minus the same under a plain system prompt (after AdiyogiV's steer-humor notebook). `terse` is the
useful one (short answers without a prompt); `pirate` and `grumpy` are for looking at."""
import json, os, sys, urllib.request
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(H, "out", "steer"); LAYERS = list(range(24, 32))
PLAIN = "You are a helpful assistant."
TRAITS = {"pirate": "You are a pirate captain. Always answer in thick pirate speech, full of sea slang.",
          "grumpy": "You are a grumpy old man who finds every question annoying. Answer with complaints and sarcasm.",
          "terse": "Answer in as few words as possible. No explanations, no lists, no pleasantries."}
MADE_FROM = ["How do I boil an egg?", "What is a mortgage?", "Why is the sky blue?", "How does a bicycle stay upright?", "What should I pack for a weekend trip?", "Explain what a contract is.",
             "How do I change a flat tire?", "What is the difference between weather and climate?", "Give me advice for a job interview.", "How does a refrigerator work?", "What is inflation?",
             "How do I make a good cup of coffee?", "Why do cats purr?", "What is a limited partnership?", "How do vaccines work?", "Tell me about the moon.", "How do I write a polite email?",
             "What is a non-compete clause?", "How do planes fly?", "What makes bread rise?", "How should I save for retirement?", "What is photosynthesis?", "How do I learn a new language?",
             "What does a notary do?", "Why do we sleep?", "How does the internet work?", "What is a balance sheet?", "How do I grow tomatoes?", "What is an indemnity?", "Why is the sea salty?",
             "How do I prepare for a marathon?", "What is a will?"]
ASK = ["What is a confidentiality agreement?", "How do I make pancakes?", "Why do leaves change colour in autumn?", "What is the capital of Canada?", "My printer will not print. What should I do?",
       "Summarize what a lease is.", "How many minutes are in an hour?", "Give me one tip for a first meeting with a client."]


def chat(system, q):
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def prompts():
    os.makedirs(OUT, exist_ok=True); idx = []
    with open(os.path.join(OUT, "in.jsonl"), "w") as f:
        for name, system in list(TRAITS.items()) + [("plain", PLAIN)]:
            for q in MADE_FROM:
                f.write(json.dumps({"text": chat(system, q)}) + "\n"); idx.append(name)
    json.dump(idx, open(os.path.join(OUT, "index.json"), "w")); print(len(idx), "texts")


def vectors():
    sys.path.insert(0, os.path.expanduser("~/models/llama.cpp/src/gguf-py")); import gguf
    m = json.load(open(os.path.join(OUT, "meta.json"))); L, d = m["shape"]["l_out"]; X = np.fromfile(os.path.join(OUT, "l_out.f32"), dtype=np.float32).reshape(-1, L, d)
    idx = np.array(json.load(open(os.path.join(OUT, "index.json")))); rep = {}
    for name in TRAITS:
        diff = X[idx == name].mean(0) - X[idx == "plain"].mean(0); path = os.path.join(OUT, name + ".gguf")          # [layer, hidden]
        if os.path.exists(path):
            os.remove(path)
        w = gguf.GGUFWriter(path, "controlvector"); w.add_string("controlvector.model_hint", "qwen35"); w.add_uint32("controlvector.layer_count", len(LAYERS))
        for i, layer in enumerate(LAYERS):
            w.add_tensor(f"direction.{layer}", diff[i].astype(np.float32))
        w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
        rep[name] = {"size_by_layer": [round(float(np.linalg.norm(diff[i])), 2) for i in range(L)], "stream_size_by_layer": [round(float(np.linalg.norm(X[idx == "plain"][:, i], axis=1).mean()), 2) for i in range(L)]}
    json.dump(rep, open(os.path.join(OUT, "vectors.json"), "w"), indent=1); print(json.dumps(rep))


def demo(trait, scale):
    for q in ASK:
        body = {"model": "local", "temperature": 0, "max_tokens": 160, "chat_template_kwargs": {"enable_thinking": False}, "messages": [{"role": "system", "content": PLAIN}, {"role": "user", "content": q}]}
        r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8082/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=300))
        a = (r["choices"][0]["message"].get("content") or "").strip()
        open(os.path.join(OUT, "demo.jsonl"), "a").write(json.dumps({"trait": trait, "scale": float(scale), "q": q, "answer": a, "words": len(a.split()), "cut": r["choices"][0]["finish_reason"] == "length"}, ensure_ascii=False) + "\n")
        print(trait, scale, "|", q, "->", a[:160].replace("\n", " "), flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    {"prompts": prompts, "vectors": vectors, "demo": lambda: demo(a[1], a[2])}[a[0]]()
