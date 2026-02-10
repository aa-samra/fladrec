import numpy as np

from fladrec.data.sequential import TrainDataset, EvalDatasetGTS, collate_fn
from torch.utils.data import DataLoader

def seek_source_batch(users, dataset: TrainDataset, njobs=1):
    source_users = dataset._dataset['uid'].to_numpy()
    shared_users = np.intersect1d(
        source_users, users.cpu().numpy()
    )
    indices = np.where(np.isin(source_users, shared_users))[0]
    
    if njobs == 1:
        batch = [dataset.__getitem__(i) for i in indices]
    else:
        from joblib import Parallel, delayed
        batch = Parallel(n_jobs=njobs)(
            delayed(dataset.__getitem__)(i) for i in indices
        )
    if len(batch)==0:
        return None
    else:
        return collate_fn(batch)