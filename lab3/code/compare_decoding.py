import pickle
import torch

from gpt_model import GPTLike

CONTEXT_LEN = 128
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open("data/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
stoi, itos, vocab_size = vocab["stoi"], vocab["itos"], vocab["vocab_size"]


def decode(ids):
    return "".join(itos.get(int(i), "?") for i in ids)


def encode(text):
    return [stoi.get(ch, vocab["unk_id"]) for ch in text]


model = GPTLike(vocab_size=vocab_size, context_len=CONTEXT_LEN, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS)
model.load_state_dict(torch.load("out/ckpt_final.pt", map_location=DEVICE))
model.to(DEVICE)
model.eval()


def distinct_n(text: str, n: int) -> float:
    tokens = list(text)
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


PROMPTS = ["ROMEO:", "To be, or not to be", "KING RICHARD:"]
N_RUNS_PER_STRATEGY = 3
MAX_NEW_TOKENS = 200

strategies = {
    "greedy": dict(temperature=0.0, top_k=None, top_p=None),
    "temperature=0.8": dict(temperature=0.8, top_k=None, top_p=None),
    "temperature=1.2": dict(temperature=1.2, top_k=None, top_p=None),
    "top_k=20": dict(temperature=0.8, top_k=20, top_p=None),
    "top_p=0.9": dict(temperature=0.8, top_k=None, top_p=0.9),
}

results = {}

for strat_name, kwargs in strategies.items():
    all_texts = []
    n_runs = 1 if strat_name == "greedy" else N_RUNS_PER_STRATEGY
    for prompt in PROMPTS:
        prompt_ids = torch.tensor([encode(prompt)], device=DEVICE)
        for _ in range(n_runs):
            gen_ids = model.generate(prompt_ids, max_new_tokens=MAX_NEW_TOKENS, **kwargs)[0].tolist()
            gen_text = decode(gen_ids)
            all_texts.append(gen_text)

    d1 = sum(distinct_n(t, 1) for t in all_texts) / len(all_texts)
    d2 = sum(distinct_n(t, 2) for t in all_texts) / len(all_texts)
    d3 = sum(distinct_n(t, 3) for t in all_texts) / len(all_texts)

    results[strat_name] = {"distinct-1": d1, "distinct-2": d2, "distinct-3": d3, "samples": all_texts}

    print(f"\n{'=' * 20} {strat_name} {'=' * 20}")
    print(f"distinct-1: {d1:.3f}  distinct-2: {d2:.3f}  distinct-3: {d3:.3f}")
    print(f"example (prompt: {PROMPTS[0]!r}):\n{all_texts[0][:300]}")

with open("out/decoding_comparison.pkl", "wb") as f:
    pickle.dump(results, f)

with open("out/decoding_comparison.txt", "w", encoding="utf-8") as f:
    for strat_name, r in results.items():
        f.write(f"=== {strat_name} ===\n")
        f.write(f"distinct-1: {r['distinct-1']:.3f}  distinct-2: {r['distinct-2']:.3f}  distinct-3: {r['distinct-3']:.3f}\n")
        for i, s in enumerate(r["samples"]):
            f.write(f"--- sample {i} ---\n{s}\n")
        f.write("\n")

