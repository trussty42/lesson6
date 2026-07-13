"""
Слои для decoder-only (GPT-подобного) трансформера.

В отличие от EncoderLayer/DecoderLayer из layers.py (энкодер-декодер архитектура
для машинного перевода), здесь используется только один self-attention блок
с маской из будущих токенов (causal mask) - без encoder-decoder attention,
так как энкодера у нас нет вообще. Это и есть архитектура GPT.

Классы намеренно повторяют стиль EncoderLayer/Encoder из layers.py (pre-norm
residual connections, deepcopy при построении стека слоёв), чтобы сохранить
единую структуру кода урока.
"""
import torch
from copy import deepcopy
from typing import Optional

from layers import MultiheadAttention, FeedForward


class DecoderOnlyLayer(torch.nn.Module):
    """Один блок decoder-only трансформера: masked self-attention + FFN."""

    def __init__(self, mha: MultiheadAttention, ffn: FeedForward, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = deepcopy(mha)
        self.ffn = deepcopy(ffn)
        self.layernorm1 = torch.nn.LayerNorm(mha.d_model)
        self.layernorm2 = torch.nn.LayerNorm(mha.d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_norm = self.layernorm1(x)
        x = x + self.attention(x_norm, x_norm, x_norm, mask)[0]

        x_norm = self.layernorm2(x)
        x = self.dropout(x + self.ffn(x_norm))
        return x


class DecoderOnly(torch.nn.Module):
    """Стек из num_layers блоков DecoderOnlyLayer."""

    def __init__(self, layer: DecoderOnlyLayer, num_layers: int) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([deepcopy(layer) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x
