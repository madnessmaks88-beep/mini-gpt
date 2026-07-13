"""
Шаг 4: Полная GPT-модель.

Собираем всё вместе: multi-head self-attention + feed-forward + 
residual connections + layer norm = блок трансформера.
Складываем несколько таких блоков друг на друга = GPT.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

# --- Гиперпараметры ---
batch_size = 64
block_size = 256       # теперь модель видит контекст в 256 символов, а не 8 как в биграмме
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384            # размер embedding-вектора для каждого символа
n_head = 6              # количество голов attention
n_layer = 6             # количество блоков трансформера друг на друге
dropout = 0.2           # регуляризация - случайно "выключаем" часть нейронов при обучении

print(f"Используем устройство: {device}")

# --- Данные ---
checkpoint = torch.load('data.pt')
train_data = checkpoint['train_data']
val_data = checkpoint['val_data']
vocab_size = checkpoint['vocab_size']
itos = checkpoint['itos']


def get_batch(split: str):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """Одна голова self-attention (та же логика, что в 03_attention_demo.py)."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # register_buffer - это не параметр модели (не обучается),
        # но должен ехать вместе с моделью на GPU/сохраняться в чекпоинте
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # масштабируем на sqrt(head_size) - без этого при большом head_size
        # значения после матричного умножения "взрываются" и softmax
        # становится слишком "острым" (почти one-hot), что мешает обучению
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """Несколько голов attention параллельно, результаты объединяются."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)  # смешиваем результаты голов
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """Простая полносвязная сеть - даёт модели 'подумать' над результатом attention."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Один блок трансформера: attention + feed-forward, с residual-связями."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # x + ... это и есть residual connection: сигнал идёт в обход слоя
        # LayerNorm ставим ДО attention/ffwd (pre-norm) - так стабильнее обучается
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # финальная нормализация
        self.lm_head = nn.Linear(n_embd, vocab_size)  # проекция обратно в размер словаря

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B,T,n_embd) - "что за символ"
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,n_embd) - "на какой позиции"
        x = tok_emb + pos_emb  # складываем - модель знает и символ, и его позицию
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # обрезаем контекст до block_size - модель не умеет работать с большим
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == '__main__':
    model = GPTLanguageModel().to(device)
    print(f"Параметров в модели: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model)
            print(f"шаг {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("\n--- Генерация текста ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=500)[0].tolist()
    print(''.join(itos[i] for i in generated))

    torch.save(model.state_dict(), 'gpt_model.pt')
    print("\nМодель сохранена в gpt_model.pt")
