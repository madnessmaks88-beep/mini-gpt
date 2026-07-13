"""
Шаг 1: Данные и токенизация.

Задача: превратить сырой текст в последовательность чисел, с которой
может работать нейросеть, и разбить на train/val выборки.
"""
import torch

# --- Загружаем текст ---
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Длина текста (символов): {len(text):,}")

# --- Строим словарь ---
# set(text) — уникальные символы, sorted — для детерминированного порядка
# (важно: без sorted порядок символов был бы каждый раз разный между запусками)
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Размер словаря (уникальных символов): {vocab_size}")
print(f"Символы: {''.join(chars)!r}")

# --- Кодировщик/декодировщик ---
# stoi = string to int, itos = int to string
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s: str) -> list[int]:
    """Строка -> список чисел"""
    return [stoi[c] for c in s]

def decode(l: list[int]) -> str:
    """Список чисел -> строка"""
    return ''.join(itos[i] for i in l)

# Быстрая проверка, что кодирование/декодирование работает корректно
test_str = "Привет, мир!"
encoded = encode(test_str)
decoded = decode(encoded)
print(f"\nПроверка: {test_str!r} -> {encoded[:10]}... -> {decoded!r}")
assert decoded == test_str, "Ошибка кодирования!"

# --- Кодируем весь текст в тензор ---
data = torch.tensor(encode(text), dtype=torch.long)
print(f"\nРазмер тензора данных: {data.shape}, dtype: {data.dtype}")

# --- Разбиваем на train/val (90/10) ---
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"Train: {len(train_data):,} символов, Val: {len(val_data):,} символов")

# Сохраняем всё, что понадобится в следующих шагах
torch.save({
    'train_data': train_data,
    'val_data': val_data,
    'vocab_size': vocab_size,
    'stoi': stoi,
    'itos': itos,
}, 'data.pt')

print("\nСохранено в data.pt")
