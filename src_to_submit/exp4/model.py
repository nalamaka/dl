from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_pad_idx: int,
        tgt_pad_idx: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 512,
        tie_embeddings: bool = True,
    ):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.d_model = d_model

        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=src_pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=tgt_pad_idx)
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.output_proj = nn.Linear(d_model, tgt_vocab_size, bias=False)
        if tie_embeddings:
            self.output_proj.weight = self.tgt_embed.weight
        self._reset_parameters(tie_embeddings=tie_embeddings)

    def _reset_parameters(self, tie_embeddings: bool) -> None:
        std = self.d_model ** -0.5
        nn.init.normal_(self.src_embed.weight, mean=0.0, std=std)
        nn.init.normal_(self.tgt_embed.weight, mean=0.0, std=std)

        if self.src_embed.padding_idx is not None:
            with torch.no_grad():
                self.src_embed.weight[self.src_embed.padding_idx].fill_(0.0)
        if self.tgt_embed.padding_idx is not None:
            with torch.no_grad():
                self.tgt_embed.weight[self.tgt_embed.padding_idx].fill_(0.0)

        if not tie_embeddings:
            nn.init.xavier_uniform_(self.output_proj.weight)

    def _causal_mask(self, tgt_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones((tgt_len, tgt_len), device=device, dtype=torch.bool),
            diagonal=1,
        )

    def forward(self, src: torch.Tensor, tgt_input: torch.Tensor) -> torch.Tensor:
        src_pad_mask = src.eq(self.src_pad_idx)
        tgt_pad_mask = tgt_input.eq(self.tgt_pad_idx)
        tgt_mask = self._causal_mask(tgt_input.size(1), src.device)

        src_emb = self.pos_encoder(self.src_embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embed(tgt_input) * math.sqrt(self.d_model))

        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        return self.output_proj(out)

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        bos_idx: int,
        eos_idx: int,
        max_len: int,
    ) -> torch.Tensor:
        self.eval()
        batch_size = src.size(0)
        generated = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=src.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)

        for _ in range(max_len):
            logits = self.forward(src, generated)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            finished |= next_token.squeeze(1).eq(eos_idx)
            if finished.all():
                break
        return generated

    @staticmethod
    def _no_repeat_blocked_tokens(tokens: list[int], ngram_size: int) -> set[int]:
        if ngram_size <= 1 or len(tokens) < ngram_size - 1:
            return set()

        prefix = tuple(tokens[-(ngram_size - 1):]) if ngram_size > 1 else tuple()
        blocked: set[int] = set()
        for i in range(len(tokens) - ngram_size + 1):
            ngram = tokens[i : i + ngram_size]
            if tuple(ngram[:-1]) == prefix:
                blocked.add(int(ngram[-1]))
        return blocked

    @torch.no_grad()
    def beam_decode(
        self,
        src: torch.Tensor,
        bos_idx: int,
        eos_idx: int,
        max_len: int,
        beam_size: int = 4,
        length_penalty: float = 0.6,
        no_repeat_ngram_size: int = 0,
    ) -> torch.Tensor:
        self.eval()
        batch_out: list[torch.Tensor] = []

        for i in range(src.size(0)):
            src_i = src[i : i + 1]
            seq = self._beam_decode_single(
                src_i=src_i,
                bos_idx=bos_idx,
                eos_idx=eos_idx,
                max_len=max_len,
                beam_size=beam_size,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            batch_out.append(seq)

        max_out_len = max(x.size(0) for x in batch_out)
        out = torch.full((len(batch_out), max_out_len), eos_idx, dtype=torch.long, device=src.device)
        for i, seq in enumerate(batch_out):
            out[i, : seq.size(0)] = seq
        return out

    def _beam_decode_single(
        self,
        src_i: torch.Tensor,
        bos_idx: int,
        eos_idx: int,
        max_len: int,
        beam_size: int,
        length_penalty: float,
        no_repeat_ngram_size: int,
    ) -> torch.Tensor:
        beam_size = max(1, int(beam_size))
        beams: list[tuple[torch.Tensor, float, bool]] = [
            (torch.tensor([bos_idx], dtype=torch.long, device=src_i.device), 0.0, False)
        ]

        for _ in range(max_len):
            cand: list[tuple[torch.Tensor, float, bool]] = []
            all_finished = True

            for tokens, score, finished in beams:
                if finished:
                    cand.append((tokens, score, True))
                    continue
                all_finished = False

                logits = self.forward(src_i, tokens.unsqueeze(0))[:, -1, :].squeeze(0)
                log_probs = torch.log_softmax(logits, dim=-1)
                log_probs[bos_idx] = float("-inf")

                if no_repeat_ngram_size > 0:
                    blocked = self._no_repeat_blocked_tokens(tokens.tolist(), no_repeat_ngram_size)
                    if blocked:
                        blocked_ids = torch.tensor(list(blocked), dtype=torch.long, device=src_i.device)
                        log_probs[blocked_ids] = float("-inf")

                topk_vals, topk_ids = torch.topk(log_probs, k=min(beam_size, log_probs.size(0)))
                for v, idx in zip(topk_vals.tolist(), topk_ids.tolist()):
                    new_tokens = torch.cat(
                        [tokens, torch.tensor([idx], dtype=torch.long, device=src_i.device)],
                        dim=0,
                    )
                    new_finished = idx == eos_idx
                    cand.append((new_tokens, score + float(v), new_finished))

            if all_finished:
                break

            def norm_score(item: tuple[torch.Tensor, float, bool]) -> float:
                toks, raw_score, _ = item
                # Exclude BOS in length normalization.
                gen_len = max(1, toks.size(0) - 1)
                if length_penalty > 0:
                    denom = ((5.0 + gen_len) / 6.0) ** length_penalty
                    return raw_score / denom
                return raw_score

            cand.sort(key=norm_score, reverse=True)
            beams = cand[:beam_size]

        best = max(
            beams,
            key=lambda x: x[1] / ((((5.0 + max(1, x[0].size(0) - 1)) / 6.0) ** length_penalty) if length_penalty > 0 else 1.0),
        )
        return best[0]
