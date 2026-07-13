import torch
from typing import Optional
from tokenizers import Tokenizer

from layers import Embedding, MultiheadAttention, FeedForward
from gpt_layers import DecoderOnlyLayer, DecoderOnly


def get_pad_mask(x: torch.Tensor, pad_index: int) -> torch.Tensor:
    return (x != pad_index).unsqueeze(-2)


def get_subsequent_mask(x: torch.Tensor) -> torch.Tensor:
    """Треугольная маска, запрещающая "заглядывать" в будущие токены."""
    batch_size, seq_len = x.size()
    mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask


class GeneratorTransformer(torch.nn.Module):
    """Decoder-only трансформер, генерирующий текст авторегрессивно."""

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        d_ff: int = 1024,
        num_layers: int = 6,
        vocab_size: int = 32000,
        pad_index: int = 0,
        dropout: float = 0.1,
        context_len: int = 128,
        tokenizer: Optional[Tokenizer] = None,
        device: str = 'cuda',
    ) -> None:
        super().__init__()
        mha = MultiheadAttention(d_model, num_heads, dropout)
        ffn = FeedForward(d_model, d_ff, dropout)
        self.decoder = DecoderOnly(DecoderOnlyLayer(mha, ffn, dropout), num_layers)
        self.normalize = torch.nn.LayerNorm(d_model)

        self.embedding = Embedding(d_model, vocab_size, pad_index)
        self.vocab_projection = torch.nn.Linear(d_model, vocab_size)

        self.pad_index = pad_index
        self.device = device
        # self.context_len - размер окна контекста модели (max_length при обучении)
        self.context_len = context_len
        self.tokenizer = tokenizer

        if tokenizer is not None:
            self.bos_token_id = tokenizer.token_to_id('<s>')
            self.eos_token_id = tokenizer.token_to_id('</s>')
        else:
            self.bos_token_id = None
            self.eos_token_id = None

    def make_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Комбинированная маска: паддинг + запрет заглядывать в будущее."""
        pad_mask = get_pad_mask(x, self.pad_index).to(x.device)
        causal_mask = get_subsequent_mask(x).to(x.device)
        return pad_mask & causal_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.make_mask(x)
        h = self.embedding(x)
        h = self.decoder(h, mask)
        h = self.normalize(h)
        return self.vocab_projection(h)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        context_len: Optional[int] = None,
        temperature: float = 1.0,
        max_out_tokens: int = 200,
    ) -> str:
        """
        Авторегрессивная генерация продолжения текста по промпту.

        При каждой итерации к сгенерированной последовательности добавляется
        новый токен, а модель на вход получает только последние `context_len`
        токенов - контекст "скользит" по сгенерированному тексту:

            Итерация 1: [A, B, C, D] -> предсказываем E
            Итерация 2: [B, C, D, E] -> предсказываем F
            Итерация 3: [C, D, E, F] -> предсказываем G
        """
        self.eval()
        context_len = context_len or self.context_len

        input_ids = self.tokenizer.encode(prompt).ids
        input_ids = torch.tensor([input_ids], device=self.device)

        generated = input_ids.clone()
        cur_input = input_ids

        for _ in range(max_out_tokens):
            outputs = self(cur_input)
            next_token_logits = outputs[0, -1, :] / max(temperature, 1e-5)

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).view(1, 1)

            generated = torch.cat([generated, next_token], dim=1)

            # Сдвигаем контекст на 1 токен влево (оставляем последние context_len токенов)
            cur_input = generated[:, -context_len:]

            if next_token.item() == self.eos_token_id:
                break

        return self.tokenizer.decode(generated[0].tolist())

    @torch.no_grad()
    def generate_beam(
        self,
        prompt: str,
        beam_width: int = 5,
        context_len: Optional[int] = None,
        max_out_tokens: int = 200,
        length_penalty: float = 0.7,
    ) -> str:
        """Дополнительное задание: генерация с помощью beam search.

        На каждом шаге для каждого из beam_width лучших "лучей" рассматриваются
        beam_width наиболее вероятных продолжений, из полученных
        beam_width*beam_width кандидатов оставляются beam_width лучших по
        суммарному log-правдоподобию (с нормировкой на длину).
        """
        self.eval()
        context_len = context_len or self.context_len

        def norm_score(item):
            seq, score, _ = item
            length = seq.size(1)
            return score / (length ** length_penalty)

        start_ids = self.tokenizer.encode(prompt).ids
        start_tensor = torch.tensor([start_ids], device=self.device)
        beams = [(start_tensor, 0.0, False)]

        for _ in range(max_out_tokens):
            all_finished = all(finished for _, _, finished in beams)
            if all_finished:
                break

            candidates = []
            for seq, score, finished in beams:
                if finished:
                    candidates.append((seq, score, finished))
                    continue

                cur_input = seq[:, -context_len:]
                logits = self(cur_input)[0, -1, :]
                log_probs = torch.log_softmax(logits, dim=-1)
                topk_log_probs, topk_ids = log_probs.topk(beam_width)

                for lp, tid in zip(topk_log_probs, topk_ids):
                    new_seq = torch.cat([seq, tid.view(1, 1)], dim=1)
                    new_score = score + lp.item()
                    new_finished = tid.item() == self.eos_token_id
                    candidates.append((new_seq, new_score, new_finished))

            candidates.sort(key=norm_score, reverse=True)
            beams = candidates[:beam_width]

        best_seq = max(beams, key=norm_score)[0]
        return self.tokenizer.decode(best_seq[0].tolist())

    def save_checkpoint(self, path: str, config: Optional[dict] = None) -> None:
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': config or {},
        }
        torch.save(checkpoint, path)
        print(f'Checkpoint saved: {path}')

    @classmethod
    def load_from_checkpoint(cls, path: str, tokenizer: Tokenizer, device: str = 'cpu') -> 'GeneratorTransformer':
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint.get('config', {})
        model = cls(
            d_model=config.get('d_model', 256),
            num_heads=config.get('num_heads', 8),
            d_ff=config.get('d_ff', 1024),
            num_layers=config.get('num_layers', 6),
            vocab_size=config.get('vocab_size', tokenizer.get_vocab_size()),
            pad_index=config.get('pad_index', tokenizer.token_to_id('<pad>')),
            dropout=config.get('dropout', 0.1),
            context_len=config.get('context_len', 128),
            tokenizer=tokenizer,
            device=device,
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model
