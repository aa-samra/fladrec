import math
import torch 
from torch import nn
from .sasrec import SASRecPlusEncoder, create_masked_tensor
from .bert4rec import Bert4RecPlusEncoder
from .bert4rec import create_masked_tensor as create_masked_tensor_bert

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.GELU, dropout=0.0, normalize=True):
        super().__init__()
        layers = []
        prev_dim = input_dim

        if normalize:
            layers.append(nn.LayerNorm(input_dim))
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hdim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class TargetSASRec(SASRecPlusEncoder):
    def __init__(self, 
                 fusion_mode:str, 
                 cd_emb_dim=None, 
                 proj_hidden_dim=512,
                 proj_num_layers=1,
                 num_decoder_layers=1,
                 num_cd_tokens=2,
                 normalize_cd=True,
                 cd_user_emb=None,
                 cd_user_ids=None,
                 *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        self._fusion_mode = fusion_mode
        self._embedding_dim_cd = cd_emb_dim
        self._num_cd_tokens = num_cd_tokens
        
        if 'trainable' in self._fusion_mode:
            assert not cd_user_emb is None
            self.cd_user_embedding = nn.Parameter(cd_user_emb, requires_grad=True)
            self.cd_user_ids = cd_user_ids

        if 'fuse' in self._fusion_mode:
            self._fuser = MLP(input_dim=self._embedding_dim * 2,  
                              output_dim=self._embedding_dim,
                              hidden_dims=[self._embedding_dim * 2])
        
        if 'decode' in self._fusion_mode:
            decoder_layer = nn.TransformerDecoderLayer(d_model=self._embedding_dim, 
                                                   nhead=self._num_heads,
                                                   batch_first=True)
            self._decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        if 'tokenize':
            self._cd_token_embeddings = nn.Parameter(
                torch.randn((self._num_cd_tokens, self._embedding_dim))
            )
            tokenizer_layer = nn.TransformerDecoderLayer(d_model=self._embedding_dim, 
                                                   nhead=self._num_heads,
                                                   batch_first=True)
            self._tokenizer = nn.TransformerDecoder(tokenizer_layer, num_layers=num_decoder_layers)
            self._prompt_token = nn.Parameter(torch.randn(self._embedding_dim))

        self._normalize_cd = normalize_cd
        self._projector = MLP(input_dim=self._embedding_dim_cd, 
                              output_dim=self._embedding_dim,
                              hidden_dims=[proj_hidden_dim] * proj_num_layers,
                              dropout=self._dropout.p,
                              normalize=self._normalize_cd)
        
        self._init_weights(1/math.sqrt(float(self._embedding_dim)))

    def find_shared_user_index(self, user_ids, shared_user_ids):
        matching_mask = (user_ids[:,None]==shared_user_ids)
        return torch.where(matching_mask)
    
    def apply_fusion_mode(self, embeddings, shared_user_mask, proj_user_cd_emb):
        seq_len = embeddings.shape[1]
        if 'decode' in self._fusion_mode:
            embeddings[shared_user_mask] = self._decoder(
                embeddings[shared_user_mask], # [batch_size_shared, seq_len, embedding_dim]
                proj_user_cd_emb.unsqueeze(1), # [batch_size_shared, 1, embedding_dim]
            )
        elif 'fuse' in self._fusion_mode:
            embeddings[shared_user_mask] = self._fuser(
                torch.cat([
                    embeddings[shared_user_mask], # [batch_size_shared, seq_len, embedding_dim]
                    proj_user_cd_emb.unsqueeze(1).repeat([1, seq_len, 1]), # [batch_size_shared, seq_len, embedding_dim]
                ], dim=-1) 
            )
        elif 'add' in self._fusion_mode: 
            embeddings[shared_user_mask] = \
                    embeddings[shared_user_mask] +  proj_user_cd_emb.unsqueeze(1).repeat([1, seq_len, 1]) # [batch_size_shared, seq_len, embedding_dim]

        return embeddings
    
    def append_cd_tokens(self, embeddings, mask, shared_user_mask, proj_user_cd_emb):
        assert self.padding=='left', 'this method assume left-padding'
        assert ((~mask[shared_user_mask]).sum(dim=1).min() >= (self._num_cd_tokens)), "At least _num_cd_tokens should be empty before adding tokens"
        
        batch_size_shared = len(shared_user_mask)
        cd_decoded_tokens = self._tokenizer(
            tgt=self._cd_token_embeddings.unsqueeze(0).repeat([batch_size_shared, 1, 1]), #[batch_size_shared, n_cd_tokens, embedding_dim]
            memory=proj_user_cd_emb.unsqueeze(1)  #[batch_size_shared, 1, embedding_dim]
            ) #[batch_size_shared, n_cd_tokens, embedding_dim]
        
        seq_len = mask.shape[1]
        shared_user_lengths = mask[shared_user_mask].sum(dim=1)

        cd_tokens_mask = torch.logical_and(
            torch.arange(end=seq_len, device=mask.device)[None] < (seq_len - shared_user_lengths[:, None]),
            torch.arange(end=seq_len, device=mask.device)[None] >= (seq_len - shared_user_lengths[:, None] - self._num_cd_tokens)
        )

        shared_user_embeddings = embeddings[shared_user_mask]  #[batch_size_shared, seq_len, embedding_dim]
        shared_user_embeddings[cd_tokens_mask] = cd_decoded_tokens.reshape(-1, self._embedding_dim)

        embeddings[shared_user_mask] = shared_user_embeddings

        mask[shared_user_mask] = torch.logical_or(mask[shared_user_mask], cd_tokens_mask)

        return embeddings, mask

        
    
    def forward(self, inputs, ret_emb=False, cd_emb=None, cd_user_ids=None):
        all_sample_events = inputs['item.ids']  # (total_batch_events)
        all_sample_lengths = inputs['item.length']  # (batch_size)

        embeddings = self._item_embeddings(all_sample_events)
        embeddings, mask = create_masked_tensor(embeddings, all_sample_lengths, max_sequence_length=self._max_sequence_length, padding=self.padding)

        batch_user_ids = inputs['user.ids'] #(batch_size)

        if 'trainable' in self._fusion_mode:
            cd_user_ids = self.cd_user_ids
            cd_emb = self.cd_user_embedding
            
        if not cd_user_ids is None:
            shared_user_mask, cd_user_mask = self.find_shared_user_index(batch_user_ids, cd_user_ids) #(batch_size)
            cd_emb = cd_emb[cd_user_mask]

        valid_items_mask = mask.clone()
        
        if not cd_user_ids is None and shared_user_mask.any():
            proj_user_cd_emb = self._projector(cd_emb) # [batch_size_shared, embedding_dim]
            if 'before' in self._fusion_mode:
                embeddings = self.apply_fusion_mode(embeddings, shared_user_mask, proj_user_cd_emb)
            
            if 'tokenize' in self._fusion_mode:
                embeddings, mask = self.append_cd_tokens(embeddings, mask, shared_user_mask, proj_user_cd_emb)

        embeddings, mask = self._apply_sequential_encoder(embeddings, mask)  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)
        
        if not cd_user_ids is None and shared_user_mask.any():
            if 'after' in self._fusion_mode:
                embeddings = self.apply_fusion_mode(embeddings, shared_user_mask, proj_user_cd_emb)

        if not self.training or ret_emb: # eval mode
            last_embeddings = self._get_last_embedding(embeddings, valid_items_mask)  # (batch_size, embedding_dim)
            return last_embeddings

        else:  # training mode
            return self.calc_loss(inputs, embeddings, valid_items_mask)

class TargetBERT4Rec(Bert4RecPlusEncoder):
    def __init__(self, 
                 fusion_mode:str, 
                 cd_emb_dim=None, 
                 proj_hidden_dim=512,
                 proj_num_layers=1,
                 num_decoder_layers=1,
                 num_cd_tokens=2,
                 normalize_cd=True,
                 cd_user_emb=None,
                 cd_user_ids=None,
                 *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        self._fusion_mode = fusion_mode
        self._embedding_dim_cd = cd_emb_dim
        self._num_cd_tokens = num_cd_tokens
        
        if 'trainable' in self._fusion_mode:
            raise NotImplementedError

        if 'fuse' in self._fusion_mode:
            raise NotImplementedError
        
        if 'decode' in self._fusion_mode:
            raise NotImplementedError

        if 'tokenize' in self._fusion_mode:
            raise NotImplementedError

        self._normalize_cd = normalize_cd
        self._projector = MLP(input_dim=self._embedding_dim_cd, 
                              output_dim=self._embedding_dim,
                              hidden_dims=[proj_hidden_dim] * proj_num_layers,
                              dropout=self._dropout.p,
                              normalize=self._normalize_cd)
        
        self._init_weights(1/math.sqrt(float(self._embedding_dim)))

    def find_shared_user_index(self, user_ids, shared_user_ids):
        matching_mask = (user_ids[:,None]==shared_user_ids)
        return torch.where(matching_mask)
    
    def apply_fusion_mode(self, embeddings, shared_user_mask, proj_user_cd_emb):
        seq_len = embeddings.shape[1]
        if 'add' in self._fusion_mode: 
            embeddings[shared_user_mask] = \
                    embeddings[shared_user_mask] +  proj_user_cd_emb.unsqueeze(1).repeat([1, seq_len, 1]) # [batch_size_shared, seq_len, embedding_dim]

        return embeddings
    
    def append_cd_tokens(self, embeddings, mask, shared_user_mask, proj_user_cd_emb):
        assert self.padding=='left', 'this method assume left-padding'
        assert ((~mask[shared_user_mask]).sum(dim=1).min() >= (self._num_cd_tokens)), "At least _num_cd_tokens should be empty before adding tokens"
        
        batch_size_shared = len(shared_user_mask)
        cd_decoded_tokens = self._tokenizer(
            tgt=self._cd_token_embeddings.unsqueeze(0).repeat([batch_size_shared, 1, 1]), #[batch_size_shared, n_cd_tokens, embedding_dim]
            memory=proj_user_cd_emb.unsqueeze(1)  #[batch_size_shared, 1, embedding_dim]
            ) #[batch_size_shared, n_cd_tokens, embedding_dim]
        
        seq_len = mask.shape[1]
        shared_user_lengths = mask[shared_user_mask].sum(dim=1)

        cd_tokens_mask = torch.logical_and(
            torch.arange(end=seq_len, device=mask.device)[None] < (seq_len - shared_user_lengths[:, None]),
            torch.arange(end=seq_len, device=mask.device)[None] >= (seq_len - shared_user_lengths[:, None] - self._num_cd_tokens)
        )

        shared_user_embeddings = embeddings[shared_user_mask]  #[batch_size_shared, seq_len, embedding_dim]
        shared_user_embeddings[cd_tokens_mask] = cd_decoded_tokens.reshape(-1, self._embedding_dim)

        embeddings[shared_user_mask] = shared_user_embeddings

        mask[shared_user_mask] = torch.logical_or(mask[shared_user_mask], cd_tokens_mask)

        return embeddings, mask

        
    
    def forward(self, inputs, ret_emb=False, cd_emb=None, cd_user_ids=None):
        all_sample_events = inputs['item.ids']  # (total_batch_events)
        all_sample_lengths = inputs['item.length']  # (batch_size)

        embeddings = self._item_embeddings(all_sample_events)
        embeddings, mask = create_masked_tensor_bert(embeddings, all_sample_lengths, 
                                                     mask_token_id= self._num_items,
                                                     max_sequence_length=self._max_sequence_length, 
                                                     padding=self.padding)

        batch_user_ids = inputs['user.ids'] #(batch_size)

        if not cd_user_ids is None:
            shared_user_mask, cd_user_mask = self.find_shared_user_index(batch_user_ids, cd_user_ids) #(batch_size)
            cd_emb = cd_emb[cd_user_mask]

        valid_items_mask = mask.clone()
        
        if not cd_user_ids is None and shared_user_mask.any():
            proj_user_cd_emb = self._projector(cd_emb) # [batch_size_shared, embedding_dim]
            if 'before' in self._fusion_mode:
                embeddings = self.apply_fusion_mode(embeddings, shared_user_mask, proj_user_cd_emb)
            
        embeddings, mask = self._apply_sequential_encoder(embeddings, mask)  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)
        
        if not cd_user_ids is None and shared_user_mask.any():
            if 'after' in self._fusion_mode:
                raise NotImplementedError
        
        last_embeddings = self._get_last_embedding(embeddings, valid_items_mask)  # (batch_size, embedding_dim)
        if not self.training or ret_emb: # eval mode
            return last_embeddings

        else:  # training mode
            return self.calc_loss(inputs, last_embeddings)