import numpy as np
import polars as pl
import torch
from tqdm import tqdm
from fladrec.models.sasrec import SASRecEncoder
from torch.utils.data import DataLoader

from fladrec.data.sequential import EvalDatasetGTS, collate_fn
from .metrics import Targets, Ranked, calc_metrics

def infer_users_(eval_dataloader: DataLoader, model: torch.nn.Module, device: str):
    user_ids = []
    user_embeddings = []

    model.eval()
    for batch in eval_dataloader:
        for key in batch.keys():
            batch[key] = batch[key].to(device)

        user_ids.append(batch['user.ids'])  # (batch_size)
        user_embeddings.append(model(batch))  # (batch_size, embedding_dim)

    return torch.cat(user_ids, dim=0), torch.cat(user_embeddings, dim=0)

def infer_users(eval_dataloader: DataLoader, model: torch.nn.Module, device: str):
    user_ids = []
    user_embeddings = []
    targets = []

    model.eval()
    for batch in tqdm(eval_dataloader):
        for key in batch.keys():
            batch[key] = batch[key].to(device)

        user_ids.append(batch['user.ids'])  # (batch_size)
        user_embeddings.append(model(batch))  # (batch_size, embedding_dim)
        targets.append(batch['labels.ids'])

    return torch.cat(user_ids, dim=0), torch.cat(user_embeddings, dim=0), torch.cat(targets, dim=0)


def sample_excluding_numpy(N, S, k=999):
    S = np.array(list(S))
    population = np.setdiff1d(np.arange(1, N + 1), S, assume_unique=False)
    return np.random.choice(population, size=k, replace=False)

def recommend(
    eval_dataloader,
    model,
    device,
    k=100,
    downvote_seen=True,
    return_uid=False,
    sample_metric=None, 
):
    item_embedding = infer_items(model)  # [num_items, dim]
    num_items = item_embedding.size(0)
    model.eval()

    ranked_items = []
    target_items = []
    uids = []

    if sample_metric:
        k = sample_metric+1

    for seed, batch in tqdm(enumerate(eval_dataloader)):
        for key in batch.keys():
            batch[key] = batch[key].to(device)

        with torch.no_grad():
            user_embeddings = model(batch)  # [B, dim]

            # full scores
            scores = user_embeddings @ item_embedding.T  # [B, num_items]

            if downvote_seen:
                seen_idx = batch['seen.ids']
                lengths = batch['seen.length']
                user_idx = torch.repeat_interleave(
                    torch.arange(len(lengths), device=device), lengths
                )
                scores[user_idx, seen_idx] = -torch.inf
                scores[:, 0] = -torch.inf  # padding item

            # ---------- NEW: sampled metric ----------
            if sample_metric is not None:
                seen_idx = batch['seen.ids']
                lengths = batch['seen.length']
                user_idx = torch.repeat_interleave(
                    torch.arange(len(lengths), device=device), lengths
                )
                torch.manual_seed(int(seed))
                random_scores = torch.rand_like(scores)
                random_scores[user_idx, seen_idx] = -1

                sampled_items = torch.topk(random_scores, k=sample_metric+1, dim=1).indices
                sampled_items[:, -1] = batch['labels.ids']
                # gather sampled scores
                sampled_scores = scores.gather(1, sampled_items)

                # top-k within sampled set
                topk_idx = torch.topk(sampled_scores, k=k, dim=1).indices
                recs = sampled_items.gather(1, topk_idx)
            else:
                recs = torch.topk(scores, k=k).indices
            # ----------------------------------------

        ranked_items.append(recs)
        target_items.append(batch['labels.ids'])
        uids.append(batch['user.ids'])

    if return_uid:
        return (
            torch.cat(ranked_items, dim=0),
            torch.cat(target_items, dim=0),
            torch.cat(uids, dim=0),
        )
    else:
        return torch.cat(ranked_items, dim=0), torch.cat(target_items, dim=0)

    
        


def infer_items(model: SASRecEncoder):
    return model.item_embeddings.weight.data

def get_eval_dataloader(train_df, eval_df, max_seq_len, batch_size, seed=42, eval_mode='random', no_train=False):
    if no_train:
        eval_df = eval_df.with_columns(
            pl.lit([]).alias("item_id_train")
        )
        eval_df = eval_df.with_columns(
            pl.col('item_id').alias("item_id_valid")
        )
    else:
        eval_df = train_df.join(eval_df, on='uid', how='right', suffix='_valid').select(
            pl.col('uid'), pl.col('item_id').alias('item_id_train'), pl.col('item_id_valid')
            ).sort('uid')
        eval_df = eval_df.with_columns(
            pl.col("item_id_train").fill_null([])
        )

    eval_dataset = EvalDatasetGTS(dataset=eval_df, max_seq_len=max_seq_len, seed=seed, mode=eval_mode)

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=8,
        shuffle=False,
    )

    return eval_df, eval_dataloader


def eval_model(eval_dataloader, model, device, downvote_seen, sample_metric=None):
    model.eval()
    with torch.inference_mode():
        ranked_items, target_items = recommend(eval_dataloader=eval_dataloader, k=100, downvote_seen=downvote_seen, model=model, device=device, sample_metric=sample_metric)
    ranked = Ranked(user_ids=torch.arange(len(ranked_items), device=device), 
                    item_ids=ranked_items, 
                    scores=None, 
                    num_item_ids=model._num_items)
    target = Targets(user_ids=torch.arange(len(ranked_items), device=device),
                     item_ids=[item.reshape(-1) for item in target_items])
    
    # metric_names = [f'{name}@{k}' for name in ["recall", "ndcg"] for k in [5, 10]]
    metric_names = [f'{name}@{k}' for name in ["recall", "ndcg", "coverage"] for k in [10, 50, 100]]

    if sample_metric:
        metric_names.append(f'mrr@{sample_metric+1}')
    metrics = calc_metrics(ranked, target, metrics=metric_names)

    return metrics
