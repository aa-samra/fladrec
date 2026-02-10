from typing import Dict, Tuple

import torch
import torch.nn as nn


def create_masked_tensor(data: torch.Tensor, lengths: torch.Tensor, max_sequence_length=None, padding='right') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts a batch of variable-length sequences into a padded tensor and corresponding mask.

    Args:
        data (torch.Tensor): Input tensor containing flattened sequences.
            - For indices: shape (total_elements,) of dtype long
            - For embeddings: shape (total_elements, embedding_dim)
        lengths (torch.Tensor): 1D tensor of sequence lengths, shape (batch_size,)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - padded_tensor: Padded tensor of shape:
                - (batch_size, max_seq_len) for indices
                - (batch_size, max_seq_len, embedding_dim) for embeddings
            - mask: Boolean mask of shape (batch_size, max_seq_len) where True indicates valid elements

    Note:
        - Zero-padding is added to the right of shorter sequences
    """
    batch_size = lengths.shape[0]
    max_sequence_length = (max_sequence_length if not max_sequence_length is None 
                           else int(lengths.max().item()))

    if len(data.shape) == 1:  # indices
        padded_tensor = torch.zeros(
            batch_size, max_sequence_length, dtype=data.dtype, device=data.device
        )  # (batch_size, max_seq_len)
    else:
        assert len(data.shape) == 2  # embeddings
        padded_tensor = torch.zeros(
            batch_size, max_sequence_length, *data.shape[1:], dtype=data.dtype, device=data.device
        )  # (batch_size, max_seq_len, embedding_dim)

    if padding=='left':
        mask = (
            torch.arange(end=max_sequence_length, device=lengths.device)[None] >= (max_sequence_length - lengths[:, None])
        )  # (batch_size, max_seq_len)
    elif padding=='right':
        mask = (
            torch.arange(end=max_sequence_length, device=lengths.device)[None] < lengths[:, None]
        )  # (batch_size, max_seq_len)

    assert mask.sum() == lengths.sum()

    padded_tensor[mask] = data

    return padded_tensor, mask



class SASRecEncoder(nn.Module):
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
        padding: str='left',
    ) -> None:
        super().__init__()
        self._num_items = num_items
        self._num_heads = num_heads
        self._embedding_dim = embedding_dim
        self.padding = padding 

        self._item_embeddings = nn.Embedding(
            num_embeddings=num_items + 1,  # add zero id embedding
            embedding_dim=embedding_dim,
        )
        self._position_embeddings = nn.Embedding(num_embeddings=max_sequence_length, embedding_dim=embedding_dim)

        self._layernorm = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self._dropout = nn.Dropout(dropout)

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward or 4 * embedding_dim,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
        )
        self._encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers)

        self._init_weights(initializer_range)

    @property
    def item_embeddings(self) -> nn.Module:
        return self._item_embeddings

    @property
    def num_items(self) -> int:
        return self._num_items

    def _apply_sequential_encoder(self, embeddings: torch.Tensor, mask: torch.Tensor):
        """"
        Processes variable-length event sequences through a transformer encoder with positional embeddings.
        Args:
            - embedding: (batch_size, max_seq_len, embedding_dim) for embeddings
            - mask: Boolean mask of shape (batch_size, max_seq_len) where True indicates valid elements

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - embeddings: Processed sequence embeddings, shape (batch_size, seq_len, embedding_dim)
                - mask: Boolean mask indicating valid elements, shape (batch_size, seq_len)

        Processing Steps [CHANGES]:
            1. Embedding Lookup was moved outside this function
            2. Positional Encoding:
                - Generates inverse-order positions (newest event First - 0)
                - Adds positional embeddings to item embeddings
            3. Transformer Processing:
                - Applies layer norm and dropout
                - Uses causal attention mask for autoregressive modeling
                - Uses padding mask to ignore invalid positions

        Note:
            - Position indices are generated in chronological order (newest event = position -1)
        """
        batch_size = mask.shape[0]
        seq_len = mask.shape[1]
        lengths = mask.to(torch.int).sum(dim=1)

        if self.padding=='left':
            positions = (
                torch.arange(start=seq_len-1, end=-1, step=-1, device=mask.device)[None].tile([batch_size, 1]).long()
            )  # (batch_size, seq_len)
            positions_mask = mask
        elif self.padding=='right':
            positions = (
                torch.arange(start=seq_len-1, end=-1, step=-1, device=mask.device)[None].tile([batch_size, 1]).long()
            )  # (batch_size, seq_len)
            positions_mask = torch.flip(mask, dims=[1])  # (batch_size, max_seq_len)

        positions = positions[positions_mask]  # (total_batch_events)
        position_embeddings = self._position_embeddings(positions)  # (total_batch_events, embedding_dim)
        position_embeddings, _ = create_masked_tensor(
            data=position_embeddings, lengths=lengths, max_sequence_length=seq_len, padding=self.padding
        )  # (batch_size, seq_len, embedding_dim)

        embeddings = embeddings + position_embeddings  # (batch_size, seq_len, embedding_dim)
        embeddings = self._layernorm(embeddings)  # (batch_size, seq_len, embedding_dim)
        embeddings = self._dropout(embeddings)  # (batch_size, seq_len, embedding_dim)
        embeddings[~mask] = 0

        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool().to(mask.device)  # (seq_len, seq_len)
        embeddings = self._encoder(
            src=embeddings, mask=~causal_mask, src_key_padding_mask=~mask
        )  # (batch_size, seq_len, embedding_dim)

        return embeddings, mask
    
    @torch.no_grad()
    def _init_weights(self, initializer_range: float) -> None:
        """
        Initialize all model parameters (weights and biases) in-place.

        For each parameter in the model:
            - If the parameter name contains 'weight':
                - If it also contains 'norm' (e.g., for normalization layers), initialize with ones.
                - Otherwise, initialize with a truncated normal distribution (mean=0, std=initializer_range)
                and values clipped to the range [-2 * initializer_range, 2 * initializer_range].
            - If the parameter name contains 'bias', initialize with zeros.
            - If the parameter name does not match either case, raise a ValueError.

        Args:
            initializer_range (float): Standard deviation for the truncated normal distribution
                used to initialize non-normalization weights.

        Note:
            This method should be called during model initialization to ensure all weights and biases
            are properly set. It runs in a no-grad context and does not track gradients.
        """
        for key, value in self.named_parameters():
            if 'weight' in key:
                if 'norm' in key:
                    nn.init.ones_(value.data)
                else:
                    nn.init.trunc_normal_(
                        value.data, std=initializer_range, a=-2 * initializer_range, b=2 * initializer_range
                    )
            else:
                # assert 'bias' in key
                nn.init.zeros_(value.data)

    @staticmethod
    def _get_last_embedding(embeddings: torch.Tensor, mask: torch.Tensor, padding='right') -> torch.Tensor:
        """
        Extracts the embedding of the last valid (non-padded) element from each sequence in a batch.

        Args:
            embeddings (torch.Tensor): Tensor of shape (batch_size, seq_len, embedding_dim)
                containing embeddings for each element in each sequence.
            mask (torch.Tensor): Boolean tensor of shape (batch_size, seq_len) indicating
                valid (True) and padded (False) positions in each sequence.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, embedding_dim) containing the embedding
                of the last valid element for each sequence in the batch.
        """
        if padding=='left':
            last_embeddings = embeddings[:, -1, :]  # (batch_size, embedding_dim)
        elif padding=='right':
            flatten_embeddings = embeddings[mask]  # (total_batch_events, embedding_dim)
            lengths = torch.sum(mask, dim=-1)  # (batch_size)
            offsets = torch.cumsum(lengths, dim=0)  # (batch_size)
            last_embeddings = flatten_embeddings[offsets.long() - 1]  # (batch_size, embedding_dim)

        return last_embeddings

    def forward(self, inputs: Dict) -> torch.Tensor:
        """
        Forward pass of the model, handling both training and evaluation modes.

        Args:
            inputs (Dict): Input dictionary containing:
                - 'item.ids' (torch.LongTensor): Flattened tensor of item IDs for all sequences in the batch.
                    Shape: (total_batch_events,)
                - 'item.length' (torch.LongTensor): Sequence lengths for each sample in the batch.
                    Shape: (batch_size,)
                - 'positive.ids' (torch.LongTensor, training only): Positive sample IDs for contrastive learning.
                    Shape: (total_batch_events,)
                - 'negative.ids' (torch.LongTensor, training only): Negative sample IDs for contrastive learning.
                    Shape: (total_batch_events,)

        Returns:
            torch.Tensor:
                - During training: Binary cross-entropy loss between positive/negative sample scores.
                    Shape: (1,)
                - During evaluation: Embeddings of the last valid item in each sequence.
                    Shape: (batch_size, embedding_dim)
        """
        all_sample_events = inputs['item.ids']  # (total_batch_events)
        all_sample_lengths = inputs['item.length']  # (batch_size)
        
        embedding = self._item_embeddings(all_sample_events)
        embedding, mask = create_masked_tensor(embedding, all_sample_lengths, padding=self.padding)
        
        embeddings, mask = self._apply_sequential_encoder(
            embedding, mask
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)


        if self.training:  # training mode
            # queries
            in_batch_queries_embeddings = embeddings[mask]  # (total_batch_events, embedding_dim)

            # positives
            in_batch_positive_events = inputs['positive.ids']  # (total_batch_events)
            in_batch_positive_embeddings = self._item_embeddings(
                in_batch_positive_events
            )  # (total_batch_events, embedding_dim)
            positive_scores = torch.einsum(
                'bd,bd->b', in_batch_queries_embeddings, in_batch_positive_embeddings
            )  # (total_batch_events)

            # negatives
            in_batch_negative_events = inputs['negative.ids']  # (total_batch_events)
            in_batch_negative_embeddings = self._item_embeddings(
                in_batch_negative_events
            )  # (total_batch_events, embedding_dim)
            negative_scores = torch.einsum(
                'bd,bd->b', in_batch_queries_embeddings, in_batch_negative_embeddings
            )  # (total_batch_events)

            loss = nn.functional.binary_cross_entropy_with_logits(
                torch.cat([positive_scores, negative_scores], dim=0),
                torch.cat([torch.ones_like(positive_scores), torch.zeros_like(negative_scores)]),
            )  # (1)

            return loss
        else:  # eval mode
            last_embeddings = self._get_last_embedding(embeddings, mask, padding=self.padding)  # (batch_size, embedding_dim)
            return last_embeddings


class SASRecPlusEncoder(SASRecEncoder):
    def __init__(
        self,
        num_items: int,
        max_sequence_length: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        num_neg_items: int=1,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
        activation: nn.Module = nn.GELU(),
        layer_norm_eps: float = 1e-9,
        initializer_range: float = 0.02,
        sce_alpha: int | None = 16.,
        sce_beta: int | None = 1.,
        loss_only_grad_accum: bool = False,
        grad_accum_bs: int | None = None,
        loss_type: str='sce',
        padding: str='left',
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
            padding=padding
        )

        self._max_sequence_length = max_sequence_length

        self._num_neg_items = num_neg_items
        self._loss_type = loss_type
        if loss_type == 'sce':
            self._alpha = sce_alpha
            self._beta = sce_beta
        if loss_type == 'ce':
            self._loss_only_grad_accum = loss_only_grad_accum
            self._grad_accum_bs = grad_accum_bs




    def forward(self, inputs: Dict, ret_emb=False) -> torch.Tensor:
        """
        Forward pass of the model, handling both training and evaluation modes.

        Args:
            inputs (Dict): Input dictionary containing:
                - 'item.ids' (torch.LongTensor): Flattened tensor of item IDs for all sequences in the batch.
                    Shape: (total_batch_events,)
                - 'item.length' (torch.LongTensor): Sequence lengths for each sample in the batch.
                    Shape: (batch_size,)
                - 'positive.ids' (torch.LongTensor, training only): Positive sample IDs for contrastive learning.
                    Shape: (total_batch_events,)
                - 'negative.ids' (torch.LongTensor, training only): Negative sample IDs for contrastive learning.
                    Shape: (total_batch_events,)

        Returns:
            torch.Tensor:
                - During training: Binary cross-entropy loss between positive/negative sample scores.
                    Shape: (1,)
                - During evaluation: Embeddings of the last valid item in each sequence.
                    Shape: (batch_size, embedding_dim)
        """
        all_sample_events = inputs['item.ids']  # (total_batch_events)
        all_sample_lengths = inputs['item.length']  # (batch_size)
        
        embedding = self._item_embeddings(all_sample_events)
        embedding, mask = create_masked_tensor(embedding, all_sample_lengths, padding=self.padding, max_sequence_length=self._max_sequence_length)
        
        embeddings, mask = self._apply_sequential_encoder(
            embedding, mask
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)


        if not self.training or ret_emb: # eval mode
            last_embeddings = self._get_last_embedding(embeddings, mask, padding=self.padding)  # (batch_size, embedding_dim)
            return last_embeddings

        else:  # training mode
            return self.calc_loss(inputs, embeddings, mask)

    def calc_loss(self, inputs, embeddings, mask):
        if self._loss_type == 'bce':
            # queries
            in_batch_queries_embeddings = embeddings[mask]  # (total_batch_events, embedding_dim)

            # positives
            in_batch_positive_events = inputs['positive.ids']  # (total_batch_events)
            in_batch_positive_embeddings = self._item_embeddings(
                in_batch_positive_events
            )  # (total_batch_events, embedding_dim)
            positive_scores = torch.einsum(
                'bd,bd->b', in_batch_queries_embeddings, in_batch_positive_embeddings
            )  # (total_batch_events)

            # negatives
            in_batch_negative_events = inputs['negative.ids']  # (total_batch_events, n_negative_samples)


            in_batch_negative_embeddings = self._item_embeddings(
                in_batch_negative_events
            )  # (total_batch_events, n_negative_samples, embedding_dim)

            negative_scores = torch.einsum(
                'bd,bnd->bn', in_batch_queries_embeddings, in_batch_negative_embeddings
            )  # (total_batch_events, n_negative_samples)

            loss = nn.functional.binary_cross_entropy_with_logits(
                torch.cat([positive_scores[:, None], negative_scores], dim=-1),
                torch.cat([torch.ones_like(positive_scores)[:, None], torch.zeros_like(negative_scores)], dim=-1),
                reduction='none'
            ).sum(dim=-1).mean()
            print_loss = loss.item()
        elif self._loss_type == 'ce':
            # queries
            in_batch_queries_embeddings = embeddings[mask]  # (total_batch_events, embedding_dim)

            # positives
            in_batch_positive_events = inputs['positive.ids']  # (total_batch_events)

            if self._loss_only_grad_accum:
                bs = in_batch_queries_embeddings.shape[0]
                grad_buffer = torch.zeros_like(in_batch_queries_embeddings)
                print_loss = 0
                for start in range(0, bs, self._grad_accum_bs):
                    end = start + self._grad_accum_bs
                    chunk_in_batch_queries_embeddings = in_batch_queries_embeddings[start:end].detach().requires_grad_(True)
                    chunk_in_batch_positive_events = in_batch_positive_events[start:end]


                    chunk_scores = torch.einsum(
                        'bd,Cd->bC', chunk_in_batch_queries_embeddings, self.item_embeddings.weight
                    )

                    chunk_loss = nn.functional.cross_entropy(chunk_scores, chunk_in_batch_positive_events, reduction='sum')
                    chunk_loss.backward()

                    grad_buffer[start:end] = chunk_in_batch_queries_embeddings.grad.detach() / bs


                    print_loss += chunk_loss.item() / bs

                # print(grad_buffer.sum())
                # with torch.no_grad():
                #     print(grad_accum_bs * 165, bs, grad_accum_bs, bs / grad_accum_bs, grad_accum_bs / bs)
                #     # self.item_embeddings.weight.grad.data = self.item_embeddings.weight.grad.data * grad_accum_bs * 165


                loss = (in_batch_queries_embeddings * grad_buffer).sum()

            else:

                in_batch_scores = torch.einsum(
                    'bd,Cd->bC', in_batch_queries_embeddings, self.item_embeddings.weight
                ) # (total_batch_events, catalog_size)

                # print(in_batch_scores.dtype)

                loss = nn.functional.cross_entropy(in_batch_scores, in_batch_positive_events, reduction='mean')
                print_loss = loss.item()
        elif self._loss_type == 'sce':
            # queries
            in_batch_queries_embeddings = embeddings[mask]  # (total_batch_events, embedding_dim)

            # positives
            in_batch_positive_events = inputs['positive.ids']  # (total_batch_events)

            in_batch_positive_embeddings = self._item_embeddings(
                in_batch_positive_events
            )

            bs, hd = in_batch_queries_embeddings.shape
            n_buckets =  int((bs / self._beta) ** 0.5 * self._alpha)
            bucket_size_x = int((bs * self._beta) ** 0.5 * self._alpha)
            bucket_size_y = self._num_neg_items

            x = in_batch_queries_embeddings
            y = in_batch_positive_events

            w = self.item_embeddings.weight

            positive_scores_ = torch.einsum(
                'bd,bd->b', x, in_batch_positive_embeddings
            )  # (total_batch_events)

            with torch.no_grad():
                buckets = 1/(hd ** 0.25) * torch.randn(n_buckets, hd, device=x.device) # (n_buckets, embedding_dim)

                # (n_buckets, embedding_dim) x (embedding_dim, total_batch_events) -> (n_buckets, total_batch_events)
                x_bucket = buckets @ x.T
                # x_bucket[:, mask.view(-1) == 0] = float('-inf')
                _, top_x_bucket = torch.topk(x_bucket, dim=1, k=bucket_size_x) # (n_buckets, bucket_size_x)
                del x_bucket

                # (n_buckets, embedding_dim) x (embedding_dim, catalog_size) -> (n_buckets, catalog_size)
                y_bucket = buckets @ w.T

                _, top_y_bucket = torch.topk(y_bucket, dim=1, k=bucket_size_y) # (n_buckets, bucket_size_y)
                del y_bucket

            x_bucket = torch.gather(x, 0, top_x_bucket.view(-1, 1).expand(-1, hd))\
                .view(n_buckets, bucket_size_x, hd) # (n_buckets, bucket_size_x, embedding_dim)
            y_bucket = torch.gather(w, 0, top_y_bucket.view(-1, 1).expand(-1, hd))\
                .view(n_buckets, bucket_size_y, hd) # (n_buckets, bucket_size_y, embedding_dim)

            negative_scores = (x_bucket @ y_bucket.transpose(-1, -2)) # (n_buckets, bucket_size_x, bucket_size_y)
            mask = torch.index_select(y, dim=0, index=top_x_bucket.view(-1))\
                .view(n_buckets, bucket_size_x)[:, :, None] == top_y_bucket[:, None, :] # (n_buckets, bucket_size_x, bucket_size_y)
            negative_scores = negative_scores.masked_fill(mask, float('-inf')) # (n_buckets, bucket_size_x, bucket_size_y)
            positive_scores = \
                torch.index_select(positive_scores_, dim=0, index=top_x_bucket.view(-1))\
                .view(n_buckets, bucket_size_x)[:, :, None] # (n_buckets, bucket_size_x, 1)
            scores = torch.cat((negative_scores, positive_scores), dim=2) # (n_buckets, bucket_size_x, bucket_size_y + 1)

            loss_ = \
                nn.functional.cross_entropy(scores.view(-1, scores.shape[-1]),
                                            (scores.shape[-1] - 1)
                                            * torch.ones(scores.shape[0] * scores.shape[1],
                                                            dtype=torch.int64,
                                                            device=scores.device),
                                            reduction='none') # (n_buckets * bucket_size_x,)
            loss = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
            loss.scatter_reduce_(0, top_x_bucket.view(-1), loss_, reduce='amax', include_self=False)
            loss = loss[(loss != 0)].mean()
            print_loss = loss.item()

        return loss, print_loss
    