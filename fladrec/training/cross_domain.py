
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import polars as pl
import numpy as np
import torch.nn as nn


from fladrec.models.sasrec import SASRecPlusEncoder
from fladrec.models.fl_models import TargetSASRec

from fladrec.data.cd_tools import seek_source_batch
from fladrec.data.sequential import TrainDataset

from fladrec.evaluation.eval import eval_model, get_eval_dataloader, infer_users


torch.set_float32_matmul_precision('high')

def train_sasrec_cd(
    src_train_dataset: TrainDataset,
    tgt_train_dataloader: DataLoader,
    src_model: SASRecPlusEncoder,
    tgt_model: TargetSASRec,
    src_optimizer: torch.optim.Optimizer,
    tgt_optimizer: torch.optim.Optimizer,
    src_device: str = 'cpu',
    tgt_device: str = 'cpu',
    num_epochs: int = 100,
    do_autocast=False,
    grad_accum_steps = 1,
    epoch_shift:int=0, ## if called by a tuner
):
    tgt_scaler = GradScaler(tgt_device)
    src_scaler = GradScaler(src_device)
    tgt_model.train()
    global_batch_i = 0

    for epoch_num in range(num_epochs):
        tgt_model.train()
        qbar = tqdm(tgt_train_dataloader, desc=f"Epoch {epoch_num+epoch_shift}")
        for batch in qbar:
            # print(batch)
            user_ids = batch['user.ids'].clone().cpu()

            src_batch = seek_source_batch(users=user_ids, dataset=src_train_dataset)
            if not src_batch is None:
                for key in src_batch.keys():
                    src_batch[key] = src_batch[key].to(src_device)

                with autocast(src_device, enabled=do_autocast, dtype=torch.float16):
                    cd_user_emb_src = src_model(src_batch, ret_emb=True)

                cd_user_emb_tgt = cd_user_emb_src.detach().clone().to(tgt_device).requires_grad_(True)
                cd_user_ids = src_batch['user.ids'].to(tgt_device)
            else:
                cd_user_emb_tgt = None
                cd_user_ids = None

            for key in batch.keys():
                batch[key] = batch[key].to(tgt_device)
            
            with autocast(tgt_device, enabled=do_autocast, dtype=torch.float16):
                loss, print_loss = tgt_model(batch, 
                                             cd_emb=cd_user_emb_tgt, 
                                             cd_user_ids=cd_user_ids)
                
                loss = loss / grad_accum_steps
                
            tgt_scaler.scale(loss).backward()
            if not src_batch is None: 
                cd_user_emb_grad = cd_user_emb_tgt.grad.detach().clone().to(src_device)

                src_scaler.scale(cd_user_emb_src).backward(cd_user_emb_grad)
            
            if (global_batch_i + 1) % grad_accum_steps == 0:
                qbar.set_postfix(loss=f'{print_loss:.4f}', 
                                 step=global_batch_i // grad_accum_steps)
                tgt_scaler.step(tgt_optimizer)
                tgt_scaler.update()
                tgt_optimizer.zero_grad()
                if not src_batch is None: 
                    src_scaler.step(src_optimizer)
                    src_scaler.update()
                    src_optimizer.zero_grad()

            global_batch_i += 1

    return src_model.state_dict(), tgt_model.state_dict()

class TargetSASRecWrapper(nn.Module):
    """
    Wraps TargetSASRec so it can be used with eval_model, which expects model(batch).
    """
    def __init__(self, model, cd_emb, cd_user_ids):
        super().__init__()
        self.model = model
        self.cd_emb = cd_emb
        self.cd_user_ids = cd_user_ids
        self.item_embeddings = model.item_embeddings
        self._num_items = model._num_items

    def forward(self, batch):
        return self.model(batch, cd_emb=self.cd_emb, cd_user_ids=self.cd_user_ids)

def extract_source_emb(src_model, src_train_dataset, src_device):
    src_model.eval()

    train_df = src_train_dataset._dataset
    src_users = train_df.select('uid').unique().to_numpy().flatten()
    fake_items = np.ones_like(src_users, dtype=np.int64)
    fake_tmp = 100000000 * np.ones_like(src_users, dtype=np.int64) + 1

    void_eval_df = pl.DataFrame(
    {
        'uid': src_users,
        'item_id': fake_items.reshape(-1, 1),
        'timestamp': fake_tmp
    }
    )
    _, src_dataloader = get_eval_dataloader(
            train_df, void_eval_df, 
            max_seq_len=50, 
            batch_size=256,
            eval_mode='first')
    
    with torch.no_grad():
        rets = infer_users(src_dataloader, src_model, src_device)
    
    user_ids, user_embedding, targets = rets      
    return user_embedding, user_ids 


def evaluate_cd_model(src_model, tgt_model, eval_dataloader, src_train_dataset, src_device, tgt_device, eval_setup):
    tgt_model.eval()
    src_model.eval()
    
    downvote_seen = eval_setup.get('downvote_seen', False)
    sample_metric = eval_setup.get('sample_metrics', 0)

    cd_emb, cd_user_ids = extract_source_emb(src_model, src_train_dataset, src_device)
    cd_emb = cd_emb.to(tgt_device)
    cd_user_ids = cd_user_ids.to(tgt_device)

    print("intersection: ", len(np.intersect1d(
        cd_user_ids.cpu().numpy(),
        eval_dataloader.dataset._dataset['uid'].to_numpy()
    )))
    with torch.no_grad():
        wrapped_model = TargetSASRecWrapper(tgt_model, cd_emb, cd_user_ids)
        metrics = eval_model(eval_dataloader, wrapped_model, device=tgt_device, downvote_seen=downvote_seen, sample_metric=sample_metric)
    return metrics


