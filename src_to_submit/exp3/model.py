import torch
import torch.nn as nn


class PoetryModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_seq_len: int,
        pattern_cycle: int = 8,
        tie_embeddings: bool = True,
    ):
        super().__init__()

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.cycle_embed = nn.Embedding(max(2, pattern_cycle), d_model)

        self.embed_norm = nn.LayerNorm(d_model)
        self.embed_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.token_embed.weight

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=device),
            diagonal=1,
        )

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None):
        batch_size, seq_len = x.size()
        device = x.device

        if seq_len > self.pos_embed.num_embeddings:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds max_seq_len "
                f"{self.pos_embed.num_embeddings}."
            )

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        cycle_ids = positions % self.cycle_embed.num_embeddings

        hidden = (
            self.token_embed(x)
            + self.pos_embed(positions)
            + self.cycle_embed(cycle_ids)
        )
        hidden = self.embed_norm(hidden)
        hidden = self.embed_dropout(hidden)

        causal_mask = self._causal_mask(seq_len=seq_len, device=device)
        hidden = self.transformer(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=pad_mask,
        )
        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)
        return logits
