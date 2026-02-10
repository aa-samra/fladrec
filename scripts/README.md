# How to run experiments?

## 📌 Table of Contents

* [Training Single-Domain SASRec Baseline](#single-domain-sasrec-baseline)
  * [Training](#training)
    * [1. Configure Your Experiment](#1-configure-your-experiment)
    * [2. Run the Training Script](#2-run-the-training-script)
    * [3. Override Hyperparameters from CLI](#3-override-hyperparameters-from-cli)
  * [Testing / Evaluation](#testing--evaluation)
      * [Run the Testing Script](#1-run-the-evaluation-script)
      * [Optional Overrides](#3-optional-overrides)
* [Cross-Domain Fusion of Fixed Embeddings](#cross-domain-fusion-of-fixed-embeddings)
  * [Extracting User Embeddings](#extracting-user-embeddings)
  * [Training TargetSASRec with Fixed Embeddings](#training-targetsasrec-with-fixed-embeddings)
  * [Testing a Trained TargetSASRec Model](#testing-a-trained-targetsasrec-model)

---

## Single-Domain SASRec Baseline

### Training

This project uses **Hydra** for configuration management and supports dataset-specific overrides (e.g., `config/sd/dataset/mega.yaml`).
Below are the steps to launch training, control checkpoints, and use custom hyperparameters.

---

### 📁 1. Configure Your Experiment

The main configuration file is:

```
config/sd/train_sd.yaml
```

Dataset-specific overrides live in:

```
config/dataset/
```

Example `config/dataset/mega.yaml`:

```yaml
name: "mega"
data_dir: "../data/mega_zvuk-overlap50-minuser10-positives/zvuk"
downvote_seen: false

best_hp:
  embedding_dim: 256
  num_heads: 2
  num_layers: 3
  sce_alpha: 7.274166422859077
  learning_rate: 0.0001722994241656421
  num_neg_items: 119
```

---

### ▶️ 2. Run the Training Script

From the project root:

```bash
python train.py dataset=mega
```

Hydra will:

* Load `config/sd/train_sd.yaml`
* Apply `config/dataset/mega.yaml`
* Train for up to **50 epochs**
* Track validation metrics: `ndcg@10`
* Save the best model to:

```
../checkpoints/DATASET_best_sd.pth
```

---

### ⚙️ 3. Override Hyperparameters from CLI

```bash
python train.py dataset=mega num_epochs=30 batch_size=128 learning_rate=1e-4
```

---

### 📌 Example Full Command

```bash
python train.py dataset=mega num_epochs=50 batch_size=256 fastloader=true \
best_hp.embedding_dim=256 best_hp.learning_rate=1.7e-4
```

---

### **Note on `fastloader`**

Enabling `fastloader=true` transfers the full dataloader to GPU memory.
It speeds up batch preparation but increases memory use: roughly **8 × number of interactions** bytes.

---

### 🧪 Testing / Evaluation

Evaluate the best SASRec model using its checkpoint.

---

### 1. Run the Evaluation Script

```bash
python test_best_model.py dataset=mega
```

This will:

1. Load dataset from `dataset.data_dir`
2. Build SASRec model using `best_hp`
3. Load checkpoint:

```
checkpoints/<dataset.name>_best_sd.pth
```

---

### 3. Optional Overrides

```bash
python test_sasrec_sd.py dataset=mega batch_size=1024 eval_mode=random device=cuda:1 seed=42
```

Modes for `eval_mode`:

* `first`
* `last`
* `random` (seed-controlled)
* `successive` (real-world scenario simulation)

---

### 4. Check Results

Saved to `results.json`:

```json
[
  {
    "model_info": { ... },
    "eval_info": { ... },
    "metrics": { "ndcg@10": ..., "recall@50": ... }
  }
]
```

---

## Cross-Domain Fusion of Fixed Embeddings

### Extracting User Embeddings

This step produces **fixed user embeddings** from a trained Single-Domain SASRec model.

Extraction script config:

```
config/ff/extract.yaml
```

Output file format:

```
<output_dir>/<dataset>-<split>-last<max_items>.pth
```

---

### ▶️ 1. Run the Extraction Script

```bash
python extract_embeddings.py dataset=mega
```

---

### ⚙️ 2. Optional Overrides

```bash
python extract_embeddings.py dataset=mega split=test max_items=100 device=cuda:1
```

Parameters:

* `split=val/test`
* `max_items` (≤200)
* `output_dir`
* `device`

---

### 📁 3. Result File Format

```python
{
    "user.ids": Tensor[num_users],
    "user.emb": Tensor[num_users, dim]
}
```

---

## Training TargetSASRec with Fixed Embeddings

Train a **TargetSASRec** model using extracted user embeddings.

We recommend embeddings extracted with `max_items=50`.

Run training:

```bash
python train_sasrec_ff.py dataset=mega fusion_mode=add-before
```

The script loads `dataset.cd_emb_path`:

* If only a filename → loaded from `../embeddings/`
* If path → loaded directly

Best checkpoint saved to:

```
<checkpoint_dir>/<dataset>_<fusion_mode>_final.pth
```

---

### Optional Overrides

```bash
python train_ff.py dataset=mega fusion_mode=add-before num_epochs=60 batch_size=128 fastloader=true
```

---

### Example Dataset Config

```yaml
name: "mega"
data_dir: "../data/mega_zvuk-overlap50-minuser10-positives/mega"
downvote_seen: false
cd_emb_path: "zvuk_val_last50.pth"

best_hp:
  base:
    embedding_dim: 384
    num_heads: 4
    num_layers: 1
    sce_alpha: 3
    learning_rate: 0.0003116
    num_neg_items: 255
    loss_type: "sce"
    max_sequence_length: 200
    padding: "left"

  add-before:
    dropout: 0.27
    proj_hidden_dim: 256
    proj_num_layers: 3
    sce_alpha: 3.07
    learning_rate: 0.000264
```

---

## Testing a Trained TargetSASRec Model

```bash
python test_sasrec_ff.py dataset=mega fusion_mode=add-before
```

### Optional Overrides

```bash
python test_sasrec_ff.py dataset=mega batch_size=1024 eval_mode=random device=cuda:1 seed=15
```

## Source-Tuned Embeddings (Testing with pre-trained weights)


To evaluate a **Source + Target SASRec** model, run the script `test_sasrec_cd.py` with your Hydra configuration:

```bash
python test_sasrec_cd.py data_pair=mega_zvuk fusion_mode=add-before eval_mode=random seed=42
```



For optional overrides: `fusion_mode` can be only `add-before` right now. Use `batch_size`, `src_device`, and `tgt_device` to adapt memory limitations. Also by changing `eval_mode` and `seed` (only for `random`) you can get insights about the robustness of recommendations. 


==== Final averaged results ====

Model: single-domain
  [all]
    {'recall@10': 0.09021332114934921, 'recall@50': 0.1308036655187607, 'recall@100': 0.15289326906204223, 'ndcg@10': 0.059059173613786695, 'ndcg@50': 0.06804960370063781, 'ndcg@100': 0.0716248944401741, 'coverage@10': 0.13203563558882758, 'coverage@50': 0.23171249782185294, 'coverage@100': 0.29542864831361787}
  [shared]
    {'recall@10': 0.08398850560188294, 'recall@50': 0.12292060405015945, 'recall@100': 0.14422254562377929, 'ndcg@10': 0.054402999842345715, 'ndcg@50': 0.06302965435562458, 'ndcg@100': 0.06647198298406501, 'coverage@10': 0.09784611583992661, 'coverage@50': 0.18763827098134622, 'coverage@100': 0.24425994834207163}
  [nonshared]
    {'recall@10': 0.09865121692419052, 'recall@50': 0.14148935973644255, 'recall@100': 0.16464665830135344, 'ndcg@10': 0.0653707429766655, 'ndcg@50': 0.07485428154468536, 'ndcg@100': 0.07860980927944183, 'coverage@10': 0.07809390626137323, 'coverage@50': 0.16167299340269584, 'coverage@100': 0.2134623825730589}

Model: source-tuned
  [all]
    {'recall@10': 0.10570587813854218, 'recall@50': 0.15688536465168, 'recall@100': 0.18342675864696503, 'ndcg@10': 0.06247111484408378, 'ndcg@50': 0.07377208322286606, 'ndcg@100': 0.07806907147169113, 'coverage@10': 0.13190196904494594, 'coverage@50': 0.2130503543486844, 'coverage@100': 0.2594198519080568}
  [shared]
    {'recall@10': 0.09881577789783477, 'recall@50': 0.14813257455825807, 'recall@100': 0.17362482845783234, 'ndcg@10': 0.058516895533615075, 'ndcg@50': 0.06944449510839552, 'ndcg@100': 0.07357389918410177, 'coverage@10': 0.0962280007146528, 'coverage@50': 0.16954696746785716, 'coverage@100': 0.2116810442753884}
  [nonshared]
    {'recall@10': 0.11504559516906739, 'recall@50': 0.16875000596046447, 'recall@100': 0.19671352803707123, 'ndcg@10': 0.06783116608858109, 'ndcg@50': 0.07963824570178986, 'ndcg@100': 0.08416240066289901, 'coverage@10': 0.07440197455924229, 'coverage@50': 0.14280086552395743, 'coverage@100': 0.18261893785829142}