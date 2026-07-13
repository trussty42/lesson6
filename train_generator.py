"""
Обучение GeneratorTransformer на тексте книги.

Параметры взяты согласно рекомендациям из HOMEWORK.md:
    batch_size = 1
    max_length = 128-192 (размер контекста)
    learning_rate = 1e-4
    num_epochs = 2-4
"""
import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from tokenizers import Tokenizer

from generator_transformer import GeneratorTransformer
from text_dataset import create_dataloader


class GeneratorTrainer:
    def __init__(self, model, train_loader, dataset, config, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.dataset = dataset
        self.config = config
        self.device = device

        self.criterion = nn.CrossEntropyLoss(ignore_index=dataset.get_pad_token_id())
        self.optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 1e-4))

        self.save_dir = config.get('save_dir', 'checkpoints_gen')
        os.makedirs(self.save_dir, exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        use_amp = self.device.startswith('cuda')
        scaler = GradScaler(enabled=use_amp)

        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for i, batch in enumerate(progress_bar):
            ids = batch['ids'].to(self.device)
            # обычный language-modeling сдвиг: предсказываем следующий токен
            inputs = ids[:, :-1]
            targets = ids[:, 1:]

            self.optimizer.zero_grad()
            with autocast(device_type='cuda' if use_amp else 'cpu', dtype=torch.float16, enabled=use_amp):
                logits = self.model(inputs)
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            scaler.step(self.optimizer)
            scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{(total_loss / (i + 1)):.4f}'})

        return total_loss / (i + 1)

    def train(self):
        num_epochs = self.config.get('num_epochs', 3)
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch + 1}/{num_epochs}')
            avg_loss = self.train_epoch(epoch)
            print(f'Epoch {epoch + 1} avg loss: {avg_loss:.4f}')
            self.save_checkpoint(f'epoch_{epoch + 1}.pt')
        self.save_checkpoint('final_model.pt')

    def save_checkpoint(self, filename):
        model_config = {
            'd_model': self.config['d_model'],
            'num_heads': self.config['num_heads'],
            'd_ff': self.config['d_ff'],
            'num_layers': self.config['num_layers'],
            'vocab_size': self.dataset.get_vocab_size(),
            'pad_index': self.dataset.get_pad_token_id(),
            'dropout': self.config['dropout'],
            'context_len': self.config['max_length'],
        }
        self.model.save_checkpoint(os.path.join(self.save_dir, filename), config=model_config)


def main():
    config = {
        'batch_size': 1,
        'max_length': 160,
        'd_model': 256,
        'num_heads': 8,
        'num_layers': 6,
        'd_ff': 1024,
        'dropout': 0.1,
        'learning_rate': 1e-4,
        'num_epochs': 3,
        'save_dir': 'checkpoints_gen',
        'text_path': 'data/input.txt',
        'tokenizer_path': 'mistral_tokenizer.json',
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    tokenizer = Tokenizer.from_file(config['tokenizer_path'])

    train_loader, dataset = create_dataloader(
        text_path=config['text_path'],
        max_length=config['max_length'],
        batch_size=config['batch_size'],
        tokenizer=tokenizer,
    )

    model = GeneratorTransformer(
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        num_layers=config['num_layers'],
        vocab_size=dataset.get_vocab_size(),
        pad_index=dataset.get_pad_token_id(),
        dropout=config['dropout'],
        context_len=config['max_length'],
        tokenizer=dataset.tokenizer,
        device=device,
    )

    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    trainer = GeneratorTrainer(model, train_loader, dataset, config, device)
    trainer.train()

    # быстрая проверка генерации после обучения
    print('\n--- Пример генерации ---')
    print(model.generate('Once upon a time', context_len=config['max_length'], temperature=0.8, max_out_tokens=60))


if __name__ == '__main__':
    main()
