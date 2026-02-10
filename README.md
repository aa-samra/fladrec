# fl-recsys-adaptation

## Installation

### Preparing the environment
``` 
conda create -n fladrec_env python=3.11  
conda activate fladrec_env
pip install -r requirements-dev.txt   
pip install -e . 
```

### getting the data
#### MegaMarket/Zvuk
Open `/data/preprocess_cd_sber.ipynb` and run cells sequentially. Feel free to adjust preprocessing parameters:

```python
OVERLAP_RATIO = 0.5 # some unshared users will be dropped to achieve this ratio
MIN_USER_HISTORY = 10 # users with less interactions will be dropped
INTERACTION = 'positives'  # purchase in MegaMarket, play_duration > 30in Zvuk
# INTERACTION = 'all'  #  uncomment to include all interaction
EVAL_DAYS = 7
TEST_DAYS = 7
```

#### TenRec
to be implemented

### Running Models

Refer to `scripts/README.md`


