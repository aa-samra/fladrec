# train_source.py
from mpi4py import MPI
import torch
from torch.amp import autocast, GradScaler

from fladrec.data.cd_tools import seek_source_batch
from fladrec.models.sasrec import SASRecPlusEncoder

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

assert rank == 1, "train_source.py must be run as MPI worker (rank=1)."

def train_source(
    src_train_dataset,
    src_model: SASRecPlusEncoder,
    src_optimizer: torch.optim.Optimizer,
    src_device="cpu",
    do_autocast=False,
    grad_accum_steps=1,
):

    scaler = GradScaler(src_device)
    src_model.train()

    while True:
        # ---------------------------------------------------
        # 1. RECEIVE USER IDS FROM MASTER
        # ---------------------------------------------------
        user_ids_np = comm.recv(source=0, tag=10)
        if user_ids_np is None:
            continue

        user_ids = torch.tensor(user_ids_np, dtype=torch.long)

        # Build source batch
        src_batch = seek_source_batch(user_ids, dataset=src_train_dataset)

        if src_batch is None:
            comm.send(None, dest=0, tag=11)
            grad = comm.recv(source=0, tag=12)
            continue

        for k in src_batch.keys():
            src_batch[k] = src_batch[k].to(src_device)

        # ---------------------------------------------------
        # 2. SOURCE FORWARD
        # ---------------------------------------------------
        with autocast(src_device, enabled=do_autocast, dtype=torch.float16):
            cd_user_emb_src = src_model(src_batch, ret_emb=True)

        # SEND EMBEDDINGS TO MASTER
        comm.send(cd_user_emb_src.detach().cpu().numpy(), dest=0, tag=11)

        # ---------------------------------------------------
        # 3. RECEIVE GRADIENT FROM MASTER
        # ---------------------------------------------------
        grad_np = comm.recv(source=0, tag=12)

        if grad_np is not None:
            grad_t = torch.tensor(grad_np, device=src_device)
            scaler.scale(cd_user_emb_src).backward(grad_t)

        # ---------------------------------------------------
        # 4. OPTIMIZATION (same grad_accum logic)
        # ---------------------------------------------------
        if (train_source.step + 1) % grad_accum_steps == 0:
            scaler.step(src_optimizer)
            scaler.update()
            src_optimizer.zero_grad()

        train_source.step += 1

train_source.step = 0
