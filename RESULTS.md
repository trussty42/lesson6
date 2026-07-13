## Как запустить

```bash
pip install -r requirements.txt
# положите свой текст в data/input.txt (или другой .txt файл)
python train_generator.py          # обучение, чекпоинты в checkpoints_gen/
python chat.py                     # чат с генерацией (sampling)
python chat.py --beam              # чат с beam search
```
