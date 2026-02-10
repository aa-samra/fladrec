
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from fladrec.models.sasrec import SASRecPlusEncoder
from fladrec.models.fl_models import TargetSASRec

from fladrec.data.cd_tools import seek_source_batch
from fladrec.data.sequential import TrainDataset


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
