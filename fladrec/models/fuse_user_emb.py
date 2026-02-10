from typing import Dict, Tuple

import torch
import torch.nn as nn

from .sasrec import SASRecPlusEncoder, SASRecEncoder, create_masked_tensor

def diversity_loss(Y, M=2):
    """
    Encourages diversity among the last M embeddings in Y.
    Y: [batch, K, d]
    """
    Y = Y[:, -M: , :]                # Take first M embeddings
    Y = nn.functional.normalize(Y.reshape(-1, Y.size(2)), dim=-1)   # Normalize for cosine similarity
    sim = torch.matmul(Y, Y.T)  # [M*batch, M*batch]
    mask = ~torch.eye(sim.shape[0], dtype=bool, device=Y.device)
    loss = sim.masked_select(mask).pow(2).mean()     # average off-diagonal similarities
    return loss

def alignment_loss(embeddings1, embeddings2, mode='procrustes'):
    assert embeddings1.shape == embeddings2.shape
    embeddings1 = nn.functional.normalize(embeddings1, dim=-1)
    embeddings2 = nn.functional.normalize(embeddings2, dim=-1)

    if mode == 'procrustes':
        # Center the embeddings
        X = embeddings1 
        Y = embeddings2 

        # Compute optimal orthogonal matrix via SVD
        # YᵀX = UΣVᵀ → R = UVᵀ
        U, _, Vt = torch.linalg.svd(Y.T @ X)
        R = U @ Vt

        # Reconstruct embeddings2 to embeddings1 space
        Y_aligned = (Y @ R.T)

        # Compute reconstruction error (Frobenius norm)
        loss = torch.norm(X - Y_aligned, p='fro') / embeddings1.shape[0]

        return loss

    else:
        raise ValueError(f"Unknown mode: {mode}")

class FusionLayer(nn.Module):
    def __init__(self, 
                 primary_dim: int,
                 secondary_dim: int,
                 hidden_dim: int,
                 n_layers: int=2,
                 activation=nn.GELU(),
                 skip_connection=True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.primary_dim = primary_dim
        self.secondary_dim = secondary_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.skip_connection = skip_connection

        self.first_layer = nn.Linear(self.primary_dim + self.secondary_dim, self.hidden_dim)
        self.last_layer = nn.Linear(self.hidden_dim, self.primary_dim)

        self.middle_layers = []
        for i in range(self.n_layers-2):
            self.middle_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        
        self.init_params()

    def init_params(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)


    def forward(self, primary_emb, secondary_emb):
        assert primary_emb.shape[-1]==self.primary_dim
        assert secondary_emb.shape[-1]==self.secondary_dim
        x = torch.cat([primary_emb, secondary_emb], dim=-1)
        x = self.activation(self.first_layer(x))
        for layer in self.middle_layers:
            x = self.activation(layer(x))
        x = self.last_layer(x)
        if self.skip_connection:
            x = x + primary_emb
            
        return x

class SASRecPlusUserFusion(SASRecPlusEncoder):
    def __init__(self, loss_type, num_neg_items, num_items, max_sequence_length, embedding_dim, num_heads, num_layers, 
                 dim_feedforward = None, dropout = 0, activation = nn.GELU(), layer_norm_eps = 1e-9, initializer_range = 0.02, 
                 sce_alpha = 16, sce_beta = 1, loss_only_grad_accum = False, grad_accum_bs = None,
                 shared_user_emb=None, shared_user_ids=None, fuse_hidden_dim=512, skip_connection=True
                 ):
        super().__init__(loss_type, num_neg_items, num_items, max_sequence_length, 
                         embedding_dim, num_heads, num_layers, dim_feedforward, dropout, 
                         activation, layer_norm_eps, initializer_range, sce_alpha, sce_beta, loss_only_grad_accum, grad_accum_bs)
        
        self.fuse_hidden_dim = fuse_hidden_dim
        self.skip_connection = skip_connection
        if not shared_user_ids is None:
            assert (shared_user_ids!=-1).all(), "id=-1 is used for left padding"
            assert (len(shared_user_emb) == len(shared_user_ids)), "number of ids and embeddings mismatch"
            self.shared_user_ids = torch.cat(
                [torch.tensor([-1]).to(shared_user_ids.device), shared_user_ids])
            self.shared_user_emb = shared_user_emb
            self.cd_embedding_dim = shared_user_emb.shape[1]
            self.fusion_mlp = FusionLayer(primary_dim=self._embedding_dim, 
                                          secondary_dim=self.cd_embedding_dim,
                                          hidden_dim=self.fuse_hidden_dim,
                                          skip_connection=self.skip_connection)


    def find_shared_user_index(self, user_ids):
        return (user_ids[:,None]==self.shared_user_ids).long().argmax(dim=1) - 1
    
    def forward(self, inputs: Dict) -> torch.Tensor:
        """
        Forward pass of the model, handling both training and evaluation modes.

        Args:
            inputs (Dict): Input dictionary containing
                - 'user.ids' (torch.LongTensor): user IDs for all sequences in the batch.
                    Shape: (total_batch_size,)
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

        embeddings, mask = self._apply_sequential_encoder(
            all_sample_events, all_sample_lengths
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        seq_len = embeddings.shape[1]
        batch_user_ids = inputs['user.ids'] #(batch_size)
        batch_user_shared_index = self.find_shared_user_index(batch_user_ids) #(batch_size)
        shared_user_mask = batch_user_shared_index >= 0

        batch_user_shared_emb = self.shared_user_emb[batch_user_shared_index[shared_user_mask]]
        num_shared_users = shared_user_mask.long().sum()

        # fused_embeddings = embeddings
        fused_embeddings = torch.empty_like(embeddings)
        fused_embeddings[~shared_user_mask, :, :] = embeddings[~shared_user_mask, :, :]


        fused_embeddings[shared_user_mask, :, :] = self.fusion_mlp(
                embeddings[shared_user_mask, :, :],
                batch_user_shared_emb.unsqueeze(1).expand(num_shared_users, seq_len, -1)
        )
        
        if not self.training: # eval mode
            last_embeddings = self._get_last_embedding(fused_embeddings, mask)  # (batch_size, embedding_dim)
            return last_embeddings

        else:  # training mode
            return self.calc_loss(inputs, fused_embeddings, mask)


class HierarchicalTransformerDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, K, n_layers, decay_alpha=0.1, use_prompt_token=False):
        super().__init__()
        self.decay_alpha = decay_alpha
        self.z_proj = MLP(input_dim=input_dim, output_dim=output_dim, hidden_dims=[int(input_dim+output_dim)])
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=output_dim, nhead=8), 
            num_layers=n_layers
        )
        self.use_prompt_token = use_prompt_token
        self.K = K-1 if self.use_prompt_token else K
        self.query_embeds = nn.Parameter(torch.randn(self.K, output_dim))
        self.pos_embed = nn.Parameter(torch.randn(self.K, output_dim))
        self.prompt_token = nn.Parameter(torch.randn(output_dim))

    def forward(self, z):
        self.decay = torch.exp(
            -self.decay_alpha * torch.arange(self.K-1, -1, -1).float()
            ).to(self.query_embeds.device)
        batch_size = z.shape[0]
        # print(z.shape)
        z_cond = self.z_proj(z).unsqueeze(0)  # [1 ,B, d2]
        queries = (self.query_embeds + self.pos_embed) * self.decay[:, None] # [K, d]
        queries = queries.unsqueeze(1).repeat(1, batch_size, 1) # [K, B, d]
        out = self.decoder(queries, z_cond)
        # print(out.shape)
        if self.use_prompt_token:
            out = torch.concat([out, self.prompt_token[None, None, :].repeat(1, batch_size, 1)],
                               dim=0) # [K, B, d]
        return out.transpose(0, 1) # batch first
    
    def _init_weights(self, initializer_range: float) -> None:
        for key, value in self.named_parameters():
            if 'weight' in key:
                if 'norm' in key:
                    nn.init.ones_(value.data)
                else:
                    nn.init.trunc_normal_(
                        value.data, std=initializer_range, a=-2 * initializer_range, b=2 * initializer_range
                    )
            else:
                assert 'bias' in key
                nn.init.zeros_(value.data)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.GELU, dropout=0.0):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hdim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class SASRecPlusCDDecoder(SASRecPlusEncoder):
    def __init__(self, shared_user_emb=None, shared_user_ids=None, 
                 n_decoded_tokens=2, 
                 decoder_alpha=0.1, 
                 diversity_factor=1e-5, 
                 alignment_factor=1e-5,
                 *args,**kwargs):
        super().__init__(*args, **kwargs)
        if not shared_user_ids is None:
            assert (shared_user_ids!=-1).all(), "id=-1 is used for left padding"
            assert (len(shared_user_emb) == len(shared_user_ids)), "number of ids and embeddings mismatch"
            self.shared_user_ids = torch.cat(
                [torch.tensor([-1]).to(shared_user_ids.device), shared_user_ids])
            self.shared_user_emb = shared_user_emb
        self.decoder_alpha = decoder_alpha
        self.n_decoded_tokens = n_decoded_tokens
        self.diversity_factor = diversity_factor
        self.alignment_factor = alignment_factor
        self._decoder = HierarchicalTransformerDecoder(
            input_dim=self.shared_user_emb.shape[-1],
            output_dim=self._embedding_dim,
            K=self.n_decoded_tokens,
            decay_alpha=self.decoder_alpha,
            n_layers=4,
            use_prompt_token=True
        )


    def find_shared_user_index(self, user_ids):
        return (user_ids[:,None]==self.shared_user_ids).long().argmax(dim=1) - 1
    
    def _apply_sequential_encoder_aug(self, embeddings: torch.Tensor, mask: torch.Tensor):
        
        batch_size = mask.shape[0]
        seq_len = mask.shape[1]
        lengths = mask.long().sum(dim=1)

        positions = (
            torch.arange(start=seq_len - 1, end=-1, step=-1, device=mask.device)[None].tile([batch_size, 1]).long()
        )  # (batch_size, seq_len)
        positions_mask = positions < lengths[:, None]  # (batch_size, max_seq_len)

        positions = positions[positions_mask]  # (total_batch_events)
        position_embeddings = self._position_embeddings(positions)  # (total_batch_events, embedding_dim)
        position_embeddings, _ = create_masked_tensor(
            data=position_embeddings, lengths=lengths, 
            max_sequence_length=self._max_sequence_length
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

    def forward(self, inputs: Dict) -> torch.Tensor:
        """
        Forward pass of the model, handling both training and evaluation modes.

        Args:
            inputs (Dict): Input dictionary containing
                - 'user.ids' (torch.LongTensor): user IDs for all sequences in the batch.
                    Shape: (total_batch_size,)
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

        # embeddings, mask = self._apply_sequential_encoder(
        #     all_sample_events, all_sample_lengths, 
        # )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        batch_user_ids = inputs['user.ids'] #(batch_size)
        batch_user_shared_index = self.find_shared_user_index(batch_user_ids) #(batch_size)
        shared_user_mask = batch_user_shared_index >= 0

        batch_user_shared_emb = self.shared_user_emb[batch_user_shared_index[shared_user_mask]]

        
        embeddings = self._item_embeddings(all_sample_events)  # (total_batch_events, embedding_dim)

        embeddings, mask = create_masked_tensor(
            data=embeddings, lengths=all_sample_lengths,
            max_sequence_length=self._max_sequence_length
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        shared_user_decoded_tokens = self._decoder(batch_user_shared_emb)  # [n_batch_shared_users, K, local_dim]
        K = shared_user_decoded_tokens.shape[1]
        
        shifted_mask = mask.clone()
        shifted_mask[shared_user_mask, K:] = shifted_mask[shared_user_mask, :-K]
        shifted_mask[shared_user_mask, :K] = False
        
        augmented_mask = shifted_mask.clone()
        augmented_mask[shared_user_mask, :K] = True
        augmented_length = all_sample_lengths.clone()
        augmented_length[shared_user_mask] += K

        embeddings[shared_user_mask, K:] = embeddings[shared_user_mask, :-K]
        embeddings[shared_user_mask, :K] = shared_user_decoded_tokens

        embeddings, mask = self._apply_sequential_encoder_aug(
            embeddings, augmented_mask
        ) 

        if not self.training: # eval mode
            last_embeddings = self._get_last_embedding(embeddings, augmented_mask)  # (batch_size, embedding_dim)
            return last_embeddings

        else:  # training mode
            sce_loss, _ = self.calc_loss(inputs, embeddings, shifted_mask) 

            div_loss = self.diversity_factor * diversity_loss(shared_user_decoded_tokens, M=2)
            
            last_shared_embeddings = self._get_last_embedding(embeddings, augmented_mask)[shared_user_mask] 
            last_cd_out_embeddings = embeddings[shared_user_mask, K-1]
            align_loss = self.alignment_factor * alignment_loss(last_shared_embeddings, last_cd_out_embeddings)
            
            loss = sce_loss + div_loss + align_loss

            return loss, loss.item()
        

