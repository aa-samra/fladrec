"""
Adapted from Yambda-5B — A Large-Scale Multi-modal Dataset for Ranking And Retrieval
https://huggingface.co/datasets/yandex/yambda/tree/main/benchmarks/yambda/evaluation

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
import dataclasses
from functools import cached_property
from typing import Any, Iterable

import torch
from tqdm import tqdm


@dataclasses.dataclass
class Targets:
    user_ids: torch.Tensor
    item_ids: list[torch.Tensor]

    def __post_init__(self):
        assert len(self.item_ids) > 0
        assert self.user_ids.dim() == 1
        assert self.user_ids.shape[0] == len(self.item_ids)
        assert all(x.dim() == 1 for x in self.item_ids), "all ids should be 1D"

        assert all(x.device == self.item_ids[0].device for x in self.item_ids), "all ids should be on the same device"
        assert self.user_ids.device == self.item_ids[0].device

        if not torch.all(self.user_ids[:-1] <= self.user_ids[1:]):
            indexes = torch.argsort(self.user_ids, descending=False)
            self.item_ids = [self.item_ids[i] for i in indexes]
            self.user_ids = self.user_ids[indexes]

        assert torch.all(self.user_ids[:-1] < self.user_ids[1:]), "user_ids should be unique"

    @cached_property
    def lengths(self):
        return torch.tensor([ids.shape[0] for ids in self.item_ids], device=self.item_ids[0].device)

    def __len__(self):
        return len(self.item_ids)

    @property
    def device(self):
        return self.user_ids.device

    @classmethod
    def from_sequential(cls, df: pl.LazyFrame | pl.DataFrame, device: torch.device | str) -> 'Targets':
        df = df.lazy()
        return cls(
            df.select("uid").collect()["uid"].to_torch().to(device),
            [torch.tensor(x, device=device) for x in df.select("item_id").collect()["item_id"].to_list()],
        )


@dataclasses.dataclass
class Ranked:
    user_ids: torch.Tensor
    item_ids: torch.Tensor
    scores: torch.Tensor | None = None

    num_item_ids: int | None = None  # number of all items. Useful for coverage and etc.

    def __post_init__(self):
        if self.scores is None:
            self.scores = torch.arange(
                self.item_ids.shape[1], 0, -1, device=self.item_ids.device, dtype=torch.float32
            ).expand((self.user_ids.shape[0], self.item_ids.shape[1]))

        assert self.user_ids.dim() == 1
        assert self.scores.dim() == 2
        assert self.scores.shape == self.item_ids.shape
        assert self.user_ids.shape[0] == self.scores.shape[0]

        assert self.user_ids.device == self.scores.device == self.item_ids.device

        assert torch.all(self.scores[:, :-1] >= self.scores[:, 1:]), "scores should be sorted"

        if not torch.all(self.user_ids[:-1] <= self.user_ids[1:]):
            indexes = torch.argsort(self.user_ids, descending=False)
            self.item_ids = self.item_ids[indexes, :]
            self.scores = self.scores[indexes, :]
            self.user_ids = self.user_ids[indexes]

    @property
    def device(self):
        return self.user_ids.device



def cut_off_ranked(ranked: Ranked, targets: Targets) -> Ranked:
    mask = torch.isin(ranked.user_ids, targets.user_ids)

    assert ranked.scores is not None

    ranked = Ranked(
        user_ids=ranked.user_ids[mask],
        scores=ranked.scores[mask, :],
        item_ids=ranked.item_ids[mask, :],
        num_item_ids=ranked.num_item_ids,
    )

    assert ranked.item_ids.shape[0] == len(targets), "Ranked doesn't contain all targets.user_ids"

    return ranked


class Metric(ABC):
    @abstractmethod
    def __call__(
        self, ranked: Ranked | None, targets: Targets | None, target_mask: torch.Tensor | None, ks: Iterable[int]
    ) -> dict[int, float]:
        pass


class Recall(Metric):
    def __call__(
        self, ranked: Ranked | None, targets: Targets, target_mask: torch.Tensor, ks: Iterable[int]
    ) -> dict[int, float]:
        assert all(0 < k <= target_mask.shape[1] for k in ks)

        values = {}

        for k in ks:
            num_positives = targets.lengths.to(torch.float32)
            num_positives[num_positives == 0] = torch.inf

            values[k] = target_mask[:, :k].to(torch.float32).sum(dim=-1) / num_positives

            values[k] = torch.mean(values[k]).item()

        return values


class Precision(Metric):
    def __call__(
        self, ranked: Ranked | None, targets: Targets | None, target_mask: torch.Tensor, ks: Iterable[int]
    ) -> dict[int, float]:
        assert all(0 < k <= target_mask.shape[1] for k in ks)

        values = {}

        for k in ks:
            values[k] = target_mask[:, :k].to(torch.float32).sum(dim=-1) / k
            values[k] = torch.mean(values[k]).item()

        return values


class Coverage(Metric):
    def __init__(self, cut_off_ranked: bool = False):
        self.cut_off_ranked = cut_off_ranked

    def __call__(
        self, ranked: Ranked, targets: Targets | None, target_mask: torch.Tensor | None, ks: Iterable[int]
    ) -> dict[int, float]:
        if self.cut_off_ranked:
            assert targets is not None
            ranked = cut_off_ranked(ranked, targets)

        assert all(0 < k <= ranked.item_ids.shape[1] for k in ks)

        assert ranked.num_item_ids is not None

        values = {}
        for k in ks:
            values[k] = ranked.item_ids[:, :k].flatten().unique().shape[0] / ranked.num_item_ids

        return values


class MRR(Metric):
    def __call__(
        self, ranked: Ranked | None, targets: Targets, target_mask: torch.Tensor, ks: Iterable[int]
    ) -> dict[int, float]:
        assert all(0 < k <= target_mask.shape[1] for k in ks)

        values = {}
        for k in ks:
            num_positives = targets.lengths.to(torch.float32)

            indexes = torch.argmax(target_mask[:, :k].to(torch.float32), dim=-1).to(torch.float32) + 1
            indexes[num_positives == 0] = torch.inf

            rr = 1 / indexes

            values[k] = torch.mean(rr).item()

        return values


class DCG(Metric):
    def __call__(
        self, ranked: Ranked | None, targets: Targets | None, target_mask: torch.Tensor, ks: Iterable[int]
    ) -> dict[int, float]:
        assert all(0 < k <= target_mask.shape[1] for k in ks)

        values = {}

        discounts = 1.0 / torch.log2(
            torch.arange(2, target_mask.shape[1] + 2, device=target_mask.device, dtype=torch.float32)
        )

        for k in ks:
            dcg_k = torch.sum(target_mask[:, :k] * discounts[:k], dim=1)
            values[k] = torch.mean(dcg_k).item()

        return values


class NDCG(Metric):
    def __call__(
        self, ranked: Ranked | None, targets: Targets, target_mask: torch.Tensor, ks: Iterable[int]
    ) -> dict[int, float]:
        actual_dcg = DCG()(ranked, targets, target_mask, ks)

        ideal_target_mask = (
            torch.arange(target_mask.shape[1], device=targets.device)[None, :] < targets.lengths[:, None]
        ).to(torch.float32)
        assert target_mask.shape == ideal_target_mask.shape

        ideal_dcg = DCG()(ranked, targets, ideal_target_mask, ks)

        ndcg_values = {k: (actual_dcg[k] / ideal_dcg[k] if ideal_dcg[k] != 0 else 0.0) for k in ks}

        return ndcg_values


REGISTERED_METRIC_FN = {
    "recall": Recall(),
    "precision": Precision(),
    "mrr": MRR(),
    "dcg": DCG(),
    "ndcg": NDCG(),
    "coverage": Coverage(cut_off_ranked=False),
}


def _parse_metrics(metric_names: list[str]) -> dict[str, list[int]]:
    parsed_metrics = []

    for metric in metric_names:
        parts = metric.split('@')
        name = parts[0]

        assert len(parts) > 1, f"Invalid metric: {metric}, specify @k"

        value = int(parts[1])
        parsed_metrics.append((name, value))

    metrics = defaultdict(list)
    for m in parsed_metrics:
        metrics[m[0]].append(m[1])

    return metrics


def create_target_mask(ranked: Ranked, targets: Targets) -> torch.Tensor:
    ranked = cut_off_ranked(ranked, targets)

    assert ranked.device == targets.device
    assert ranked.item_ids.shape[0] == len(targets)

    target_mask = ranked.item_ids.new_zeros(ranked.item_ids.shape, dtype=torch.float32)

    for i, target in enumerate(tqdm(targets.item_ids, desc="Making target mask")):
        target_mask[i, torch.isin(ranked.item_ids[i], target)] = 1.0

    return target_mask


def calc_metrics(ranked: Ranked, targets: Targets, metrics: list[str]) -> dict[str, Any]:
    grouped_metrics = _parse_metrics(metrics)

    result = {}

    target_mask = create_target_mask(ranked, targets)

    for name, ks in grouped_metrics.items():
        result[name] = REGISTERED_METRIC_FN[name](ranked, targets, target_mask, ks=ks)

    return result
