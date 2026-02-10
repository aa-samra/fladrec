
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from fladrec.models.sasrec import SASRecPlusEncoder


torch.set_float32_matmul_precision('high')

def train_sasrec(
    train_dataloader: DataLoader,
    model: SASRecPlusEncoder,
    optimizer: torch.optim.Optimizer,
    device: str = 'cpu',
    num_epochs: int = 100,
    do_autocast=False,
    grad_accum_steps = 1,
    epoch_shift:int=0, ## if called by a tuner
):
    scaler = GradScaler(device)
    model.train()
    global_batch_i = 0

    for epoch_num in range(num_epochs):
        model.train()
        qbar = tqdm(train_dataloader, desc=f"Epoch {epoch_num+epoch_shift}")
        for batch in qbar:
            # print(batch)
            for key in batch.keys():
                batch[key] = batch[key].to(device)

            with autocast('cuda', enabled=do_autocast, dtype=torch.float16):
                loss, print_loss = model(batch)
                loss = loss / grad_accum_steps
                
            scaler.scale(loss).backward()

            if (global_batch_i + 1) % grad_accum_steps == 0:
                qbar.set_postfix(loss=f'{print_loss:.4f}', 
                                 step=global_batch_i // grad_accum_steps)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            global_batch_i += 1

    return model.state_dict()
