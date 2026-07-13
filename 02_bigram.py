"""
Шаг 2: Bigram-модель.

Простейшая возможная "языковая модель": предсказывает следующий символ,
глядя только на текущий. Это НЕ трансформер и почти не даёт связного
текста, но зато на ней видна вся механика: батчи, loss, обучение, генерация.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)  # для воспроизводимости результатов

# --- Загружаем подготовленные данные ---
checkpoint = torch.load('data.pt')
train_data = checkpoint['train_data']
val_data = checkpoint['val_data']
vocab_size = checkpoint['vocab_size']
itos = checkpoint['itos']

# --- Гиперпараметры ---
batch_size = 32      # сколько независимых последовательностей обрабатываем параллельно
block_size = 8       # длина контекста (сколько символов модель видит, предсказывая следующий)
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

print(f"Используем устройство: {device}")


def get_batch(split: str):
    """
    Достаём случайный батч данных.

    Идея: берём batch_size случайных стартовых точек в тексте,
    от каждой берём кусок длиной block_size — это вход (x),
    и тот же кусок, сдвинутый на 1 символ вправо — это цель (y).

    Пример при block_size=4 на тексте "привет":
      x = "прив", y = "риве"
      (модель должна по 'п' предсказать 'р', по 'пр' предсказать 'и', и т.д.)
    """
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


class BigramLanguageModel(nn.Module):
    """
    Модель, которая для каждого символа хранит вектор "логитов" —
    оценку вероятности каждого следующего символа. Никакого учёта
    контекста кроме текущего символа тут нет.
    """
    def __init__(self, vocab_size: int):
        super().__init__()
        # embedding-таблица размером [vocab_size, vocab_size]:
        # для каждого возможного текущего символа хранит распределение
        # "какой символ, скорее всего, будет следующим"
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx и targets — тензоры формы (batch_size, block_size) с номерами символов
        logits = self.token_embedding_table(idx)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens: int):
        """Генерируем текст, добавляя по одному символу за раз."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]  # берём предсказание только для последней позиции
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # сэмплируем следующий символ
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


@torch.no_grad()  # отключаем расчёт градиентов - тут они не нужны, экономим память и время
def estimate_loss(model):
    """Считаем средний loss на train и val, чтобы следить за прогрессом обучения."""
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


# --- Создаём модель и оптимизатор ---
model = BigramLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"Параметров в модели: {sum(p.numel() for p in model.parameters()):,}")

# --- Цикл обучения ---
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"шаг {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"\nФинальный loss: {loss.item():.4f}")

# --- Генерируем текст ---
print("\n--- Генерация текста (случайный шум, т.к. модель видит только 1 символ назад) ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)  # начинаем с символа №0
generated = model.generate(context, max_new_tokens=300)[0].tolist()
print(''.join(itos[i] for i in generated))

torch.save(model.state_dict(), 'bigram_model.pt')
print("\nМодель сохранена в bigram_model.pt")
