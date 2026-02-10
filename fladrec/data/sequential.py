
from dataclasses import dataclass, field
from functools import cached_property
import logging
from typing import Dict, List

import numpy as np
import polars as pl
import torch

from torch.utils.data import DataLoader


@dataclass
class Data:
    train: pl.LazyFrame
    validation: pl.LazyFrame | None
    test: pl.LazyFrame
    item_id_to_idx: dict[int, int]

    _train_user_ids: torch.Tensor | None = field(init=False, default=None)

    @property
    def num_items(self):
        return len(self.item_id_to_idx)

    @cached_property
    def num_train_users(self):
        return self.train.select(pl.len()).collect(engine="streaming").item()

    def train_user_ids(self, device):
        if self._train_user_ids is None or self._train_user_ids.device != device:
            self._train_user_ids = self.train.select('uid').collect(engine="streaming")['uid'].to_torch().to(device)
        return self._train_user_ids


class EvalDataset:
    def __init__(self, dataset: pl.DataFrame, max_seq_len: int):
        self._dataset = dataset
        self._max_seq_len = max_seq_len

    @property
    def dataset(self) -> pl.DataFrame:
        return self._dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Dict[str, List[int] | int]:
        sample = self._dataset.row(index, named=True)

        item_sequence = sample['item_id_train'][-self._max_seq_len :]
        next_items = sample['item_id_valid']

        return {
            'user.ids': [sample['uid']],
            'user.length': 1,
            'item.ids': item_sequence,
            'item.length': len(item_sequence),
            'labels.ids': next_items,
            'labels.length': len(next_items),
        }

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collates a batch of samples into batched tensors suitable for model input.

    This function processes a list of dictionaries, each containing keys like '{prefix}.ids'
    and '{prefix}.length' (the length of the sequence for that prefix). For each such prefix, it:
        - Concatenates all '{prefix}.ids' lists from the batch into a single flat list.
        - Collects all '{prefix}.length' values into a list.
        - Converts the resulting lists into torch.LongTensor objects.

    Args:
        batch (List[Dict]): List of sample dictionaries. Each sample must contain keys of the form
            '{prefix}.ids' (list of ints) and '{prefix}.length' (int).

    Returns:
        Dict[str, torch.Tensor]: Dictionary with keys '{prefix}.ids' and '{prefix}.length' for each prefix,
            where values are 1D torch.LongTensor objects suitable for model input.
    """
    processed_batch = {}
    for key in batch[0].keys():
        if key.endswith('.ids'):
            prefix = key.split('.')[0]
            assert '{}.length'.format(prefix) in batch[0]

            processed_batch[f'{prefix}.ids'] = []
            processed_batch[f'{prefix}.length'] = []

            for sample in batch:
                processed_batch[f'{prefix}.ids'].extend(sample[f'{prefix}.ids'])
                processed_batch[f'{prefix}.length'].append(sample[f'{prefix}.length'])

    for part, values in processed_batch.items():
        processed_batch[part] = torch.tensor(values, dtype=torch.long)

    return processed_batch

logger = logging.getLogger(__name__)

class TrainDataset:
    def __init__(self, dataset: pl.DataFrame,
                 num_items: int, num_neg_items: int, max_seq_len: int):
        self._dataset = dataset
        self._num_items = num_items
        self._max_seq_len = max_seq_len
        self._num_neg_items = num_neg_items
    
    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Dict[str, List[int] | int]:
        sample = self._dataset.row(index, named=True)

        item_sequence = sample['item_id'][:-1][-self._max_seq_len :]
        positive_sequence = sample['item_id'][1:][-self._max_seq_len :]

        ret_dict = {
            'user.ids': [sample['uid']],
            'user.length': 1,
            'item.ids': item_sequence,
            'item.length': len(item_sequence),
            'positive.ids': positive_sequence,
            'positive.length': len(positive_sequence),
        }
        if self._num_neg_items>0:
            negative_sequence = np.random.randint(1, self._num_items + 1, size=(len(item_sequence), self._num_neg_items)).tolist()
            ret_dict.update({
                'negative.ids': negative_sequence,
                'negative.length': len(negative_sequence),
            })
        return ret_dict
    
class EvalDatasetGTS(EvalDataset):
    def __init__(self, dataset:pl.DataFrame, max_seq_len, seed=42, mode='random'):
        super().__init__(dataset, max_seq_len)
        self.seed = seed
        self.mode = mode
        np.random.seed(seed=seed)

        self.val_seq_len = np.array([len(seq) for seq in dataset['item_id_valid']])

        if mode=='random':
            self.split_points = np.random.random(self.__len__())
            self.split_index = np.int32(np.ceil(self.val_seq_len * self.split_points))
        elif mode=='last':
            self.split_index = np.ones_like(self.val_seq_len)
        elif mode=='first':
            self.split_index = self.val_seq_len
        elif mode=='successive':
            self.cum_sum_len = np.cumsum(self.val_seq_len)
        else:
            raise ValueError('undefined GTS evaluatiom mode')

    def __len__(self):
        if self.mode=='successive':
            return self.val_seq_len.sum()
        else:
            return super().__len__()

    def __getitem__(self, index: int) -> Dict[str, List[int] | int]:

        if self.mode=='successive':
            user_index = np.searchsorted(self.cum_sum_len, index, side='right')
            val_index = index - self.cum_sum_len[user_index-1] if user_index>0 else index
            split_index = self.val_seq_len[user_index] - val_index  # from the end of sequence
        else:
            user_index = index
            split_index = self.split_index[index]
        sample = self._dataset.row(user_index, named=True)

        train_items = sample['item_id_train']
        holdout_items = sample['item_id_valid']

        all_items = train_items + holdout_items
        
        item_sequence = all_items[-split_index - self._max_seq_len:-split_index]
        next_items = [all_items[-split_index]]
        seen_items = all_items[:-split_index]

        return {
            'user.ids': [sample['uid']],
            'user.length': 1,
            'item.ids': item_sequence,
            'item.length': len(item_sequence),
            'labels.ids': next_items,
            'labels.length': len(next_items),
            'seen.ids': seen_items,
            'seen.length': len(seen_items)
        }   
    
class GPUSASRecDataloader:
    """
    GPU-accelerated dataloader for Sequential Recommendation (SASRec) models.
    
    Loads all data onto GPU memory and performs batching/shuffling operations
    directly on GPU for improved performance. Converts sequences into input-output
    pairs by shifting items (input: items[:-1], output: items[1:]).
    
    Args:
        base_loader: PyTorch DataLoader providing batches with keys:
            - 'positive.ids': Sequence items
            - 'positive.length': Sequence lengths
            - 'user.ids': User identifiers
        device: Target device (e.g., 'cuda:0')
    
    Yields:
        Dictionary with keys:
            - 'user.ids': User IDs (batch_size,)
            - 'item.ids': Input items, concatenated sequences (sum(lengths-1),)
            - 'item.lengths': Sequence lengths minus 1 (batch_size,)
            - 'positive.ids': Target items, concatenated sequences (sum(lengths-1),)
            - 'positive.lengths': Sequence lengths minus 1 (batch_size,)
    """
    
    def __init__(self, base_loader: DataLoader, device):
        self.device = device
        self.base_loader = base_loader
        self.batch_size = base_loader.batch_size
        
        # Load all data from base loader
        all_items, all_users, all_lengths = [], [], []
        for batch in base_loader:
            all_items.append(batch['positive.ids'])
            all_lengths.append(batch['positive.length'])
            all_users.append(batch['user.ids'])
        
        # Concatenate and move to GPU
        self.items = torch.cat(all_items).to(torch.int64).to(device)
        self.users = torch.cat(all_users).to(torch.int64).to(device)
        self.lengths = torch.cat(all_lengths).to(torch.int64).to(device)
        self.cum_len = torch.cat([
            torch.zeros(1, device=self.device),
            torch.cumsum(self.lengths, dim=0)
        ]).to(torch.long)

        
        # Map each item position to its sequence ID
        self.seq_ids = torch.searchsorted(
            self.cum_len[1:],
            torch.arange(len(self.items), device=device),
            right=True
        )
        
        self._batch_index = 0
        self._permutation = None
        self._reshuffle()
    
    def _reshuffle(self):
        """Generate new random permutation of sequence indices."""
        self._permutation = torch.randperm(len(self.lengths), device=self.device)
        self._batch_index = 0
    
    def __len__(self):
        """Return number of batches per epoch."""
        return len(self.base_loader)
    
    def __iter__(self):
        """Reset iterator and reshuffle data."""
        self._reshuffle()
        return self
    
    def __next__(self):
        """
        Get next batch of shifted sequences.
        
        Returns:
            Dictionary containing batched user IDs, input items, target items,
            and their corresponding lengths.
            
        Raises:
            StopIteration: When all batches have been consumed.
        """
        if self._batch_index >= len(self):
            raise StopIteration
        
        # Get indices for current batch
        start_idx = self._batch_index * self.batch_size
        end_idx = (self._batch_index + 1) * self.batch_size
        sampled_indices = self._permutation[start_idx:end_idx].sort().values
        
        # Create mask for items belonging to sampled sequences
        mask = torch.isin(self.seq_ids, sampled_indices)
        # assert mask.sum()==self.lengths[sampled_indices].sum(), f'A{mask.sum()}!={self.lengths[sampled_indices].sum()}'
        # Shift mask to get input (remove last item) and output (remove first item)
        input_mask = mask.clone()
        input_mask[self.cum_len[sampled_indices+1]-1] = False
        output_mask = mask.clone()
        output_mask[self.cum_len[sampled_indices]] = False
        
        # assert input_mask.sum()==output_mask.sum(), f'B{input_mask.sum()}!={output_mask.sum()}'
        # assert input_mask.sum()==mask.sum()-self.batch_size, f'C{input_mask.sum()}!={mask.sum()}-{self.batch_size}'
        # Extract masked items
        input_items = self.items[input_mask]
        output_items = self.items[output_mask]

        
        # Get metadata for sampled sequences
        sampled_lengths = self.lengths[sampled_indices] - 1
        sampled_users = self.users[sampled_indices]
        
        self._batch_index += 1
        
        return {
            'user.ids': sampled_users,
            'item.ids': input_items,
            'item.length': sampled_lengths,
            'positive.ids': output_items,
            'positive.length': sampled_lengths
        }
