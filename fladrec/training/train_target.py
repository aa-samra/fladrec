# train_target.py
from mpi4py import MPI
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from fladrec.models.fl_models import TargetSASRec
from fladrec.data.sequential import TrainDataset

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

assert rank == 0, "train_target.py must be run as MPI master (rank=0)."

def train_target(
    tgt_train_dataloader: DataLoader,
    tgt_model: TargetSASRec,
    tgt_optimizer: torch.optim.Optimizer,
    tgt_device="cpu",
    num_epochs=100,
    do_autocast=False,
    grad_accum_steps=1,
    epoch_shift=0,
):

    scaler = GradScaler(tgt_device)
    tgt_model.train()
    global_i = 0

    for epoch in range(num_epochs):
        qbar = tqdm(tgt_train_dataloader, desc=f"[MASTER] Epoch {epoch+epoch_shift}")

        for batch in qbar:
            user_ids = batch["user.ids"].clone().cpu()

            # ---------------------------------------------------
            # 1. SEND USER IDS TO SOURCE
            # ---------------------------------------------------
            comm.send(user_ids.numpy(), dest=1, tag=10)

            # ---------------------------------------------------
            # 2. RECEIVE EMBEDDINGS FROM SOURCE
            # ---------------------------------------------------
            cd_user_emb_tgt = comm.recv(source=1, tag=11)
            if cd_user_emb_tgt is not None:
                cd_user_emb_tgt = torch.tensor(cd_user_emb_tgt, device=tgt_device, requires_grad=True)
                cd_user_ids = batch["user.ids"].to(tgt_device)
            else:
                cd_user_emb_tgt = None
                cd_user_ids = None

            # Push batch to device
            for k in batch.keys():
                batch[k] = batch[k].to(tgt_device)

            # ---------------------------------------------------
            # 3. TARGET FORWARD
            # ---------------------------------------------------
            with autocast(tgt_device, enabled=do_autocast, dtype=torch.float16):
                loss, info_loss = tgt_model(batch,
                                            cd_emb=cd_user_emb_tgt,
                                            cd_user_ids=cd_user_ids)
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

            # ---------------------------------------------------
            # 4. SEND GRADIENT TO SOURCE
            # ---------------------------------------------------
            if cd_user_emb_tgt is not None:
                grad_np = cd_user_emb_tgt.grad.detach().cpu().numpy()
                comm.send(grad_np, dest=1, tag=12)
            else:
                comm.send(None, dest=1, tag=12)

            # ---------------------------------------------------
            # 5. OPTIMIZATION (every grad_accum)
            # ---------------------------------------------------
            if (global_i + 1) % grad_accum_steps == 0:
                qbar.set_postfix(loss=f"{info_loss:.4f}",
                                 step=global_i // grad_accum_steps)
                scaler.step(tgt_optimizer)
                scaler.update()
                tgt_optimizer.zero_grad()

            global_i += 1

    # Final state_dict
    return tgt_model.state_dict()


if __name__ == "__main__":
    pass
