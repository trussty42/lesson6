# Домашнее задание: Генератор текста на базе Transformer

## Задача

Создайте генератор текста на базе архитектуры Transformer, используя только декодер. Модель должна генерировать ответы авторегрессивно, предсказывая следующее слово на основе контекста.

## Требования

### 1. Архитектура модели

Создайте класс `GeneratorTransformer`, который авторегрессивно генерирует продолжение текста. Обучите его на книгах или каких-нибудь текстах, которые вы найдете в интернете

### 2. Токенизация

Используйте тот же токенизатор, что и в уроке, или создайте более простой:

```python
# Вариант 1: Использовать существующий токенизатор
tokenizer = Tokenizer.from_file("mistral_tokenizer.json")

# Вариант 2: Создать простой токенизатор на базе словаря
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()
```

### 4. Обучение

**Рекомендуемые параметры:**
- `batch_size = 1` (для экономии памяти)
- `max_length = 128-192` (размер контекста)
- `learning_rate = 1e-4`
- `num_epochs = 2-4`

Помните, что модель должна просмотреть ВЕСЬ ваш текст. Сдвигайте "окно" по тексту на `max_length` и итерируйтесь по всему тексту (подумайте, как создать датасет для этого). Варьируйте контекст, разделяя их на законченные блоки (абзацы, предложения), не забывайте про eos и bos токены!

### 5. Авторегрессивная генерация

Реализуйте метод генерации с правильным сдвигом контекста:

```python
def generate(self, prompt, context_len=50, temperature=1.0, max_out_tokens=200):
    """
    Генерирует ответ на основе промпта.
    
    При авторегрессии контекст сдвигается на 1 токен влево:
    - Изначально: [prompt_tokens]
    - После первого предсказания: [prompt_tokens, predicted_token]
    - При следующем предсказании: [prompt_tokens[1:], predicted_token, new_prediction]
    - И так далее, пока не достигнем max_length или EOS
    """
    self.eval()
    with torch.no_grad():
        # Токенизируйте промпт
        input_ids = self.tokenizer.encode(prompt).ids
        input_ids = torch.tensor([input_ids]).to(self.device)
        
        generated = input_ids.clone()
        
        for _ in range(max_out_tokens):
            # Получите предсказание для последнего токена
            outputs = self(input_ids)
            next_token_logits = outputs[0, -1, :] / temperature
            
            # Выберите следующий токен
            next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), 1)
            
            # Добавьте к результату
            generated = torch.cat([generated, next_token], dim=1)
            
            # Сдвиньте контекст на 1 токен влево для следующей итерации
            input_ids = generated[:, -self.context_len:]
            
            # Проверьте на EOS
            if next_token.item() == self.eos_token_id:
                break
    
    return self.tokenizer.decode(generated[0].tolist())
```

### 6. Сдвиг контекста при авторегрессии

**Ключевой момент:** При каждой итерации генерации контекст должен сдвигаться на 1 токен влево:

```
Итерация 1: [A, B, C, D] → предсказываем E
Итерация 2: [B, C, D, E] → предсказываем F  
Итерация 3: [C, D, E, F] → предсказываем G
```

Это обеспечивается строкой:
```python
input_ids = generated[:, -self.max_length:]
```

### 7. Тестирование

Создайте простой интерфейс для тестирования:

```python
def chat():
    model = GeneratorTransformer.load_from_checkpoint("checkpoint.pt")
    model.eval()
    
    while True:
        user_input = input("Вы: ")
        if user_input.lower() == 'quit':
            break
            
        response = model.generate(user_input, max_length=50, temperature=0.8)
        print(f"Бот: {response}")

if __name__ == "__main__":
    chat()
```

## Дополнительные задания

Реализуйте beam search для улучшения качества генерации

## Полезные ссылки

- [Документация tokenizers](https://huggingface.co/docs/tokenizers/)
- [PyTorch Autocast](https://pytorch.org/docs/stable/amp.html)
- [Transformer Architecture](https://arxiv.org/abs/1706.03762) 