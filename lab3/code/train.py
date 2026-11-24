import os
import math
import pickle
import time
import torch
import matplotlib.pyplot as plt

from gpt_model import GPTLike

SEED = 42
CONTEXT_LEN = 128
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.1

BATCH_SIZE = 64
LR = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
MAX_STEPS = 3000
EVAL_EVERY = 200
EVAL_ITERS = 50
WARMUP_STEPS = 200

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "out"
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(SEED)

train_ids = torch.load("data/train_ids.pt")
val_ids = torch.load("data/val_ids.pt")
with open("data/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

vocab_size = vocab["vocab_size"]
stoi, itos = vocab["stoi"], vocab["itos"]


def decode(ids):
    return "".join(itos.get(int(i), "?") for i in ids)


def get_batch(split: str):
    data = train_ids if split == "train" else val_ids
    ix = torch.randint(len(data) - CONTEXT_LEN, (BATCH_SIZE,))
    x = torch.stack([data[i:i + CONTEXT_LEN] for i in ix])
    y = torch.stack([data[i + 1:i + CONTEXT_LEN + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def lr_schedule(step: int) -> float:
    if step < WARMUP_STEPS:
        return LR * step / max(1, WARMUP_STEPS)
    progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    return 0.5 * LR * (1 + math.cos(math.pi * progress))


model = GPTLike(
    vocab_size=vocab_size,
    context_len=CONTEXT_LEN,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
).to(DEVICE)

print(f"device: {DEVICE}")
print(f"model parameters: {model.num_params():,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))

history = {"step": [], "train_loss": [], "val_loss": [], "val_ppl": []}
generation_samples = []

PROMPT = "ROMEO:"
prompt_ids = torch.tensor([[stoi.get(ch, vocab["unk_id"]) for ch in PROMPT]], device=DEVICE)

t0 = time.time()
for step in range(MAX_STEPS + 1):
    lr = lr_schedule(step)
    for g in optimizer.param_groups:
        g["lr"] = lr

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()

    if step % EVAL_EVERY == 0 or step == MAX_STEPS:
        losses = estimate_loss(model)
        val_ppl = math.exp(losses["val"])
        history["step"].append(step)
        history["train_loss"].append(losses["train"])
        history["val_loss"].append(losses["val"])
        history["val_ppl"].append(val_ppl)

        elapsed = time.time() - t0
        print(f"step {step:5d} | train_loss {losses['train']:.4f} | val_loss {losses['val']:.4f} "
              f"| val_ppl {val_ppl:.2f} | {elapsed:.1f}s")

        gen_ids = model.generate(prompt_ids, max_new_tokens=150, temperature=0.0)[0].tolist()
        gen_text = decode(gen_ids)
        generation_samples.append({"step": step, "text": gen_text})

        torch.save(model.state_dict(), os.path.join(OUT_DIR, f"ckpt_step{step}.pt"))

torch.save(model.state_dict(), os.path.join(OUT_DIR, "ckpt_final.pt"))

with open(os.path.join(OUT_DIR, "generation_samples.txt"), "w", encoding="utf-8") as f:
    for s in generation_samples:
        f.write(f"step{s['step']}\n{s['text']}\n\n")

with open(os.path.join(OUT_DIR, "history.pkl"), "wb") as f:
    pickle.dump(history, f)

plt.figure(figsize=(7, 4))
plt.plot(history["step"], history["train_loss"], label="train loss")
plt.plot(history["step"], history["val_loss"], label="val loss")
plt.xlabel("step")
plt.ylabel("cross-entropy loss")
plt.title("Train / Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "loss.png"), dpi=150)
plt.close()

plt.figure(figsize=(7, 4))
plt.plot(history["step"], history["val_ppl"], color="tab:red")
plt.xlabel("step")
plt.ylabel("perplexity")
plt.title("Validation Perplexity")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "perplexity.png"), dpi=150)
plt.close()
