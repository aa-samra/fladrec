from typing import Dict, Tuple

import torch
import torch.nn as nn

from .sasrec import SASRecEncoder


# ============================================================
# Masked tensor utility (LEFT padding, MASK at rightmost index)
# ============================================================

def create_masked_tensor(
    data: torch.Tensor,
    lengths: torch.Tensor,
    max_sequence_length=None,
    padding: str = 'left',
    mask_token_id: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    assert padding == 'left'

    batch_size = lengths.shape[0]
    max_sequence_length = (
        max_sequence_length
        if max_sequence_length is not None
        else int(lengths.max().item())
    )

    if data.dim() == 1:
        padded = torch.zeros(
            batch_size, max_sequence_length,
            dtype=data.dtype, device=data.device
        )
    else:
        padded = torch.zeros(
            batch_size, max_sequence_length, *data.shape[1:],
            dtype=data.dtype, device=data.device
        )

    base_mask = (
        torch.arange(max_sequence_length, device=lengths.device)[None]
        >= (max_sequence_length - lengths[:, None])
    )

    padded[base_mask] = data

    if mask_token_id is not None:
        if padded.dim() == 2:
            padded = torch.cat(
                [padded[:, 1:], padded.new_full((batch_size, 1), mask_token_id)],
                dim=1
            )
        else:
            padded = torch.cat(
                [padded[:, 1:, :], padded.new_zeros((batch_size, 1, padded.shape[-1]))],
                dim=1
            )

        mask = torch.cat(
            [base_mask[:, 1:], torch.ones(batch_size, 1, dtype=torch.bool, device=base_mask.device)],
            dim=1
        )
    else:
        mask = base_mask

    return padded, mask

# ============================================================
# BERT4Rec Encoder
# ============================================================

class Bert4RecEncoder(nn.Module):
    def __init__(
        self,
        num_items: int,
        max_sequence_length: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
        activation: nn.Module = nn.GELU(),
        layer_norm_eps: float = 1e-9,
        initializer_range: float = 0.02,
        padding: str = 'left',
    ) -> None:
        super().__init__()

        self._num_items = num_items
        self._embedding_dim = embedding_dim
        self.padding = padding
        self._num_heads = num_heads

        self._item_embeddings = nn.Embedding(
            num_embeddings=num_items + 1,  # +1 for MASK token
            embedding_dim=embedding_dim,
        )

        self._position_embeddings = nn.Embedding(
            num_embeddings=max_sequence_length,
            embedding_dim=embedding_dim,
        )

        self._layernorm = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self._dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward or 4 * embedding_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )

        self._encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self._init_weights(initializer_range)

    @property
    def item_embeddings(self) -> nn.Embedding:
        return self._item_embeddings

    @property
    def num_items(self) -> int:
        return self._num_items

    # --------------------------------------------------------
    # Bidirectional encoder (NO causal mask)
    # --------------------------------------------------------

    def _apply_sequential_encoder(
        self,
        embeddings: torch.Tensor,
        mask: torch.Tensor,
    ):
        batch_size, seq_len, _ = embeddings.shape

        positions = (
            torch.arange(seq_len - 1, -1, -1, device=embeddings.device)[None]
            .expand(batch_size, -1)
        )

        pos_emb = self._position_embeddings(positions)
        embeddings = embeddings + pos_emb
        embeddings = self._layernorm(embeddings)
        embeddings = self._dropout(embeddings)
        embeddings[~mask] = 0.0

        embeddings = self._encoder(
            src=embeddings,
            src_key_padding_mask=~mask,
        )

        return embeddings, mask

    # --------------------------------------------------------

    @staticmethod
    def _get_last_embedding(
        embeddings: torch.Tensor,
        mask: torch.Tensor,
        padding='left',
    ) -> torch.Tensor:
        assert padding == 'left'
        return embeddings[:, -1, :]

    # --------------------------------------------------------

    @torch.no_grad()
    def _init_weights(self, initializer_range: float) -> None:
        for key, value in self.named_parameters():
            if 'weight' in key:
                if 'norm' in key:
                    nn.init.ones_(value)
                else:
                    nn.init.trunc_normal_(
                        value, std=initializer_range,
                        a=-2 * initializer_range,
                        b=2 * initializer_range,
                    )
            else:
                nn.init.zeros_(value)


# ============================================================
# BERT4Rec + SCE / CE / BCE
# ============================================================

class Bert4RecPlusEncoder(Bert4RecEncoder):
    def __init__(
        self,
        num_neg_items: int,
        num_items: int,
        max_sequence_length: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
        activation: nn.Module = nn.GELU(),
        layer_norm_eps: float = 1e-9,
        initializer_range: float = 0.02,
        sce_alpha: float = 16.0,
        sce_beta: float = 1.0,
        loss_type: str = 'sce',
        padding: str = 'left',
    ) -> None:

        super().__init__(
            num_items=num_items,
            max_sequence_length=max_sequence_length,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            initializer_range=initializer_range,
            padding=padding,
        )
        self._max_sequence_length = max_sequence_length 
        self._num_neg_items = num_neg_items
        self._loss_type = loss_type
        self._alpha = sce_alpha
        self._beta = sce_beta

    # --------------------------------------------------------

    def forward(self, inputs: Dict, ret_emb: bool = False):
        all_sample_events = inputs['item.ids']
        all_sample_lengths = inputs['item.length']

        embedding = self._item_embeddings(all_sample_events)

        embedding, mask = create_masked_tensor(
            embedding,
            all_sample_lengths,
            padding=self.padding,
            mask_token_id=self._num_items,  # MASK token
            max_sequence_length=self._max_sequence_length
        )

        embeddings, mask = self._apply_sequential_encoder(embedding, mask)

        last_embeddings = self._get_last_embedding(
            embeddings, mask, padding=self.padding
        )

        if not self.training or ret_emb:
            return last_embeddings

        return self.calc_loss(inputs, last_embeddings)

    # --------------------------------------------------------
    # Losses computed ONLY on last position
    # --------------------------------------------------------

    def calc_loss(self, inputs: Dict, last_embeddings: torch.Tensor):

        x = last_embeddings                              # (B, D)
        positive_events = inputs['positive.ids']          # (total_positive_events,)
        positive_lengths = inputs['positive.length']      # (batch_size,)

        positive_events, positive_mask = create_masked_tensor(
            positive_events,
            positive_lengths,
            padding='left',
        )

        y = positive_events[:, -1]                         # (B,)

        w = self.item_embeddings.weight                  # (C, D)

        if self._loss_type == 'sce':

            bs, hd = x.shape

            n_buckets = int((bs / self._beta) ** 0.5 * self._alpha)
            bucket_size_x = int((bs * self._beta) ** 0.5 * self._alpha)
            bucket_size_y = self._num_neg_items

            pos_emb = self._item_embeddings(y)
            pos_scores_all = torch.einsum('bd,bd->b', x, pos_emb)

            with torch.no_grad():
                buckets = (1.0 / (hd ** 0.25)) * torch.randn(
                    n_buckets, hd, device=x.device
                )

                x_scores = buckets @ x.T
                _, top_x = torch.topk(x_scores, k=bucket_size_x, dim=1)

                y_scores = buckets @ w.T
                _, top_y = torch.topk(y_scores, k=bucket_size_y, dim=1)

            x_bucket = torch.gather(
                x, 0, top_x.view(-1, 1).expand(-1, hd)
            ).view(n_buckets, bucket_size_x, hd)

            y_bucket = torch.gather(
                w, 0, top_y.view(-1, 1).expand(-1, hd)
            ).view(n_buckets, bucket_size_y, hd)

            neg_scores = x_bucket @ y_bucket.transpose(-1, -2)

            mask_pos = (
                torch.index_select(y, 0, top_x.view(-1))
                .view(n_buckets, bucket_size_x)[:, :, None]
                == top_y[:, None, :]
            )

            neg_scores = neg_scores.masked_fill(mask_pos, float('-inf'))

            pos_scores = torch.index_select(
                pos_scores_all, 0, top_x.view(-1)
            ).view(n_buckets, bucket_size_x, 1)

            scores = torch.cat([neg_scores, pos_scores], dim=2)

            targets = torch.full(
                (scores.shape[0] * scores.shape[1],),
                scores.shape[-1] - 1,
                device=scores.device,
                dtype=torch.long,
            )

            loss_per = nn.functional.cross_entropy(
                scores.view(-1, scores.shape[-1]),
                targets,
                reduction='none',
            )

            loss = torch.zeros(bs, device=x.device)
            loss.scatter_reduce_(
                0, top_x.view(-1), loss_per,
                reduce='amax', include_self=False
            )

            loss = loss[loss != 0].mean()
            return loss, loss.item()

        raise ValueError(f"Unsupported loss type: {self._loss_type}")
