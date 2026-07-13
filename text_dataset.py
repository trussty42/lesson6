"""
Датасет для обучения GeneratorTransformer на произвольном тексте (книге).

По заданию: модель должна "просмотреть ВЕСЬ текст", окно нужно сдвигать по
тексту на max_length, а сами куски должны быть законченными блоками
(абзацы/предложения), с bos/eos токенами.

Реализация:
1. Текст разбивается на абзацы (по пустой строке).
2. Слишком длинные абзацы дополнительно дробятся на предложения (nltk),
   чтобы куски были осмысленными, а не обрублены посередине.
3. Каждый блок оборачивается в <s> ... </s>.
4. Блоки последовательно "упаковываются" в окна длиной max_length: как только
   очередной блок не помещается в текущее окно, окно фиксируется как обучающий
   пример, и упаковка продолжается в новое окно - так окно сдвигается по
   тексту на max_length и весь текст оказывается охвачен без пропусков.
"""
import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer


def _ensure_nltk_punkt():
    try:
        import nltk
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt_tab', quiet=True)
    except Exception:
        pass


def split_sentences(text: str):
    _ensure_nltk_punkt()
    try:
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)
    except Exception:
        # запасной вариант, если nltk недоступен
        return re.split(r'(?<=[.!?])\s+', text)


def split_into_paragraphs(text: str):
    """Разбивает текст на абзацы по пустым строкам."""
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    return paragraphs


class TextGenDataset(Dataset):
    """Датасет из окон токенов длиной max_length, покрывающих весь текст."""

    def __init__(
        self,
        text_path: str,
        tokenizer: Tokenizer = None,
        tokenizer_path: str = None,
        max_length: int = 160,
    ) -> None:
        self.max_length = max_length

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            tokenizer_path = tokenizer_path or f"{os.path.dirname(os.path.abspath(text_path))}/mistral_tokenizer.json"
            self.tokenizer = Tokenizer.from_file(tokenizer_path)

        if self.tokenizer.token_to_id('<pad>') is None:
            self.tokenizer.add_special_tokens(['<pad>', '<s>', '</s>'])

        self.bos_id = self.tokenizer.token_to_id('<s>')
        self.eos_id = self.tokenizer.token_to_id('</s>')
        self.pad_id = self.tokenizer.token_to_id('<pad>')

        with open(text_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        token_blocks = self._tokenize_into_blocks(raw_text)
        self.samples = self._pack_blocks(token_blocks)
        print(f'Текст разбит на {len(self.samples)} обучающих окон длиной до {max_length} токенов '
              f'(из {len(token_blocks)} законченных блоков текста)')

    def _tokenize_into_blocks(self, raw_text: str):
        """Токенизирует текст, разбитый на абзацы/предложения. Каждый блок -
        это список id токенов без bos/eos (они добавляются при упаковке)."""
        max_block_len = self.max_length - 2  # место под <s> и </s>
        blocks = []
        for paragraph in split_into_paragraphs(raw_text):
            ids = self.tokenizer.encode(paragraph).ids
            if len(ids) <= max_block_len:
                if ids:
                    blocks.append(ids)
                continue

            # абзац слишком длинный - дробим на предложения
            for sentence in split_sentences(paragraph):
                sent_ids = self.tokenizer.encode(sentence).ids
                if not sent_ids:
                    continue
                if len(sent_ids) <= max_block_len:
                    blocks.append(sent_ids)
                else:
                    # предложение всё ещё слишком длинное - режем окнами max_block_len
                    for i in range(0, len(sent_ids), max_block_len):
                        blocks.append(sent_ids[i:i + max_block_len])
        return blocks

    def _pack_blocks(self, blocks):
        """Упаковывает блоки (каждый оборачивается в bos/eos) в окна длиной
        max_length, сдвигая окно по тексту, пока не будет пройден весь текст."""
        samples = []
        buffer = []
        for ids in blocks:
            block_with_specials = [self.bos_id] + ids + [self.eos_id]

            if len(buffer) + len(block_with_specials) > self.max_length:
                if buffer:
                    samples.append(buffer)
                buffer = []

            buffer.extend(block_with_specials)

            while len(buffer) >= self.max_length:
                samples.append(buffer[:self.max_length])
                buffer = buffer[self.max_length:]

        if buffer:
            samples.append(buffer)
        return samples

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_pad_token_id(self):
        return self.pad_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.long)


class GenCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        ids = pad_sequence(batch, batch_first=True, padding_value=self.pad_token_id)
        return {'ids': ids}


def create_dataloader(
    text_path: str,
    max_length: int = 160,
    batch_size: int = 1,
    tokenizer: Tokenizer = None,
    shuffle: bool = True,
):
    dataset = TextGenDataset(text_path, tokenizer=tokenizer, max_length=max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=GenCollator(dataset.get_pad_token_id()),
    )
    return loader, dataset
