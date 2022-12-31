import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size: int, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.pow(
            1e4, -torch.arange(0, hidden_size, 2).float() / hidden_size
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        transformer_layers: int,
        emb_size: int,
        max_len: int,
        num_classes: int,
        d_model: int = 512,
        n_head: int = 8,
    ):
        super().__init__()
        self.positional_encoding = PositionalEncoding(
            hidden_size=emb_size, max_len=max_len
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=transformer_layers
        )
        self.linear = nn.Linear(d_model, num_classes)

    def forward(self, embeddings: torch.Tensor, attention_mask: torch.Tensor):
        embeddings = self.positional_encoding(embeddings)
        transformer_output = self.transformer_encoder(
            embeddings.swapaxes(1, 0), src_key_padding_mask=attention_mask.bool()
        ).swapaxes(1, 0)
        pooled_output = transformer_output.mean(dim=1)
        return self.linear(pooled_output)
