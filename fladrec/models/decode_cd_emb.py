from typing import Dict, Tuple

import torch
import torch.nn as nn

from .sasrec import SASRecPlusEncoder, SASRecEncoder, create_masked_tensor

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

class CDEmbDecoder(SASRecPlusEncoder):
    def __init__(self, shared_user_emb=None, shared_user_ids=None, 
                 num_decoder_layers=1,
                 projector_dims=512,
                 *args,**kwargs):
        super().__init__(*args, **kwargs)
        if not shared_user_ids is None:
            assert (shared_user_ids!=-1).all(), "id=-1 is used for left padding"
            assert (len(shared_user_emb) == len(shared_user_ids)), "number of ids and embeddings mismatch"
            self.shared_user_ids = torch.cat(
                [torch.tensor([-1]).to(shared_user_ids.device), shared_user_ids])
            self.shared_user_emb = shared_user_emb
        decoder_layer = nn.TransformerDecoderLayer(d_model=self._embedding_dim, 
                                                   nhead=self._num_heads,
                                                   batch_first=True)
        self._decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        if isinstance(projector_dims, list):
            self.projector_dims = projector_dims
        else: 
            self.projector_dims = [projector_dims]
        self._proj = MLP(input_dim=self.shared_user_emb.shape[1], 
                         output_dim=self._embedding_dim,
                         hidden_dims=self.projector_dims
                         )
    
    def init_params(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)

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

        batch_user_ids = inputs['user.ids'] #(batch_size)
        batch_user_shared_index = self.find_shared_user_index(batch_user_ids) #(batch_size)
        shared_user_mask = batch_user_shared_index >= 0

        if shared_user_mask.any():
            shared_user_cd_emb = self.shared_user_emb[batch_user_shared_index[shared_user_mask]] # [batch_size_shared, embedding_dim_cd]

            proj_user_cd_emb = self._proj(shared_user_cd_emb) # [batch_size_shared, embedding_dim]

            embeddings[shared_user_mask] = self._decoder(
                embeddings[shared_user_mask], # [batch_size_shared, seq_len, embedding_dim]
                proj_user_cd_emb.unsqueeze(1), # [batch_size_shared, 1, embedding_dim]
            )
        
        if not self.training: # eval mode
            last_embeddings = self._get_last_embedding(embeddings, mask)  # (batch_size, embedding_dim)
            return last_embeddings

        else:  # training mode
            return self.calc_loss(inputs, embeddings, mask)