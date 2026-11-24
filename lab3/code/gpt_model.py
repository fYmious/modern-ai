import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, context_len: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model should be divided by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(context_len, context_len)).view(1, 1, context_len, context_len)
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        att_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att_scores = att_scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att_weights = F.softmax(att_scores, dim=-1)
        att_weights = self.attn_dropout(att_weights)

        out = att_weights @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.resid_dropout(self.out_proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, expansion: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, expansion * d_model),
            nn.GELU(),
            nn.Linear(expansion * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, context_len: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, context_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPTLike(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.context_len = context_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, context_len, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.context_len, f"sequence length {T} > {self.context_len}"

        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        h = self.token_emb(idx) + self.pos_emb(pos)
        h = self.drop(h)

        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        logits = self.head(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int = None, top_p: float = None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if temperature == 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-5)

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")

                if top_p is not None:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    probs = F.softmax(sorted_logits, dim=-1)
                    cumprobs = torch.cumsum(probs, dim=-1)
                    sorted_mask = cumprobs - probs > top_p
                    sorted_logits[sorted_mask] = float("-inf")
                    logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)

                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_id], dim=1)
        return idx

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    model = GPTLike(vocab_size=65, context_len=64, d_model=128, n_heads=4, n_layers=2)
    print(f"parameters: {model.num_params():,}")
    x = torch.randint(0, 65, (2, 32))
    y = torch.randint(0, 65, (2, 32))
    logits, loss = model(x, y)
    print("logits:", logits.shape, "loss:", loss.item())
