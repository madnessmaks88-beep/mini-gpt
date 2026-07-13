"""
Шаг 5: Генерация текста из уже обученной модели.

Загружаем сохранённые веса (gpt_model.pt) и генерируем текст без
переобучения. Плюс добавляем "температуру" - параметр, управляющий
случайностью генерации.

Модель дублируется здесь (а не импортируется из 04_gpt_model.py),
потому что имя файла начинается с цифры - Python не любит такие
имена как имена модулей для import. Для реального проекта архитектуру
стоит вынести в отдельный model.py без цифры в начале - сделаем
это на шаге с деплоем.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

# --- Те же гиперпараметры, что были при обучении ---
# ВАЖНО: они должны СОВПАДАТЬ с теми, что были при обучении,
# иначе веса не встанут на место (несовпадение размеров тензоров)
block_size = 256
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
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
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        """
        temperature управляет "случайностью" генерации:
        - temperature < 1.0 (например 0.5): модель более "уверенная",
          выбирает наиболее вероятные символы чаще -> текст более
          предсказуемый и связный, но может зацикливаться на повторах
        - temperature = 1.0: как обучили, без изменений
        - temperature > 1.0 (например 1.5): больше случайности и
          "творчества", но выше риск бессмыслицы
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # делим на temperature ДО softmax
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def main():
    # --- Загружаем словарь (нужен, чтобы декодировать числа обратно в текст) ---
    checkpoint = torch.load('data.pt')
    vocab_size = checkpoint['vocab_size']
    itos = checkpoint['itos']
    stoi = checkpoint['stoi']

    # --- Загружаем обученную модель ---
    model = GPTLanguageModel(vocab_size).to(device)
    model.load_state_dict(torch.load('gpt_model.pt', map_location=device))
    model.eval()  # переводим в режим инференса (отключает dropout)
    print(f"Модель загружена. Параметров: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # --- Генерация с разной температурой для сравнения ---
    for temperature in [0.5, 0.8, 1.0, 1.3]:
        print(f"\n{'='*60}")
        print(f"Температура = {temperature}")
        print('='*60)
        torch.manual_seed(42)  # фиксируем seed, чтобы честно сравнить эффект температуры
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated = model.generate(context, max_new_tokens=300, temperature=temperature)[0].tolist()
        print(''.join(itos[i] for i in generated))

    # --- Генерация с затравкой (prompt), а не с пустого места ---
    print(f"\n{'='*60}")
    print("Генерация с затравкой текста")
    print('='*60)
    prompt = "Иван Иванович сказал"
    encoded_prompt = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)
    generated = model.generate(encoded_prompt, max_new_tokens=300, temperature=0.8)[0].tolist()
    print(''.join(itos[i] for i in generated))


if __name__ == '__main__':
    main()
