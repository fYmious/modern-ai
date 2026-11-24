import os
import pickle
import urllib.request
import torch

SEED = 42
torch.manual_seed(SEED)

OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

RAW_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
RAW_PATH = os.path.join(OUT_DIR, "tinyshakespeare_full.txt")

if not os.path.exists(RAW_PATH):
    print(f"downloading from {RAW_URL} ...")
    urllib.request.urlretrieve(RAW_URL, RAW_PATH)
else:
    print("data downloaded, using cache.")

with open(RAW_PATH, "r", encoding="utf-8") as f:
    full_text = f.read()

print(f"total symbols: {len(full_text):,}")

n = len(full_text)
train_text = full_text[: int(n * 0.90)]
val_text = full_text[int(n * 0.90): int(n * 0.95)]
test_text = full_text[int(n * 0.95):]

print(f"train: {len(train_text):,} symbols")
print(f"val:   {len(val_text):,} symbols")
print(f"test:  {len(test_text):,} symbols")

chars = sorted(list(set(train_text)))
vocab_size = len(chars)
print(f"vocab size: {vocab_size}")
print(f"chars: {''.join(chars)!r}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

UNK_ID = vocab_size
vocab_size_total = vocab_size + 1


def encode(text: str) -> list[int]:
    return [stoi.get(ch, UNK_ID) for ch in text]


def decode(ids) -> str:
    return "".join(itos.get(int(i), "?") for i in ids)


train_ids = encode(train_text)
val_ids = encode(val_text)
test_ids = encode(test_text)

torch.save(torch.tensor(train_ids, dtype=torch.long), os.path.join(OUT_DIR, "train_ids.pt"))
torch.save(torch.tensor(val_ids, dtype=torch.long), os.path.join(OUT_DIR, "val_ids.pt"))
torch.save(torch.tensor(test_ids, dtype=torch.long), os.path.join(OUT_DIR, "test_ids.pt"))

with open(os.path.join(OUT_DIR, "vocab.pkl"), "wb") as f:
    pickle.dump({"stoi": stoi, "itos": itos, "vocab_size": vocab_size_total, "unk_id": UNK_ID}, f)

with open(os.path.join(OUT_DIR, "train_raw.txt"), "w", encoding="utf-8") as f:
    f.write(train_text)
with open(os.path.join(OUT_DIR, "val_raw.txt"), "w", encoding="utf-8") as f:
    f.write(val_text)

