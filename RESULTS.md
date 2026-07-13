## Как запустить

```bash
pip install -r requirements.txt
# положить свой текст в data/input.txt
python train_generator.py          # обучение, чекпоинты в checkpoints_gen/
python chat.py                     # чат с генерацией (sampling)
python chat.py --beam              # чат с beam search
```
