"""
Простой интерфейс для тестирования обученной модели, согласно HOMEWORK.md.
"""
import argparse
import torch
from tokenizers import Tokenizer

from generator_transformer import GeneratorTransformer


def chat(checkpoint_path: str, tokenizer_path: str = 'mistral_tokenizer.json',
         temperature: float = 0.8, max_out_tokens: int = 100, use_beam: bool = False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = Tokenizer.from_file(tokenizer_path)
    if tokenizer.token_to_id('<pad>') is None:
        tokenizer.add_special_tokens(['<pad>', '<s>', '</s>'])

    model = GeneratorTransformer.load_from_checkpoint(checkpoint_path, tokenizer=tokenizer, device=device)
    model.eval()

    print('Модель загружена. Введите "quit" для выхода.')
    while True:
        user_input = input('Вы: ')
        if user_input.lower() == 'quit':
            break

        if use_beam:
            response = model.generate_beam(user_input, beam_width=5, max_out_tokens=max_out_tokens)
        else:
            response = model.generate(user_input, temperature=temperature, max_out_tokens=max_out_tokens)
        print(f'Бот: {response}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints_gen/final_model.pt')
    parser.add_argument('--tokenizer', type=str, default='mistral_tokenizer.json')
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--max_out_tokens', type=int, default=100)
    parser.add_argument('--beam', action='store_true', help='использовать beam search вместо сэмплирования')
    args = parser.parse_args()

    chat(args.checkpoint, args.tokenizer, args.temperature, args.max_out_tokens, args.beam)
