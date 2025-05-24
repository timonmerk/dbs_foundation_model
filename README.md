## Code for "Pretrained Transformer for Symptom Decoding Without Patient-Individual Training from Chronic Invasive Electrophysiology"

Installation:
I recommend using the package installer [uv](https://docs.astral.sh/uv/getting-started/installation/). The `pyproject.toml` can then simply be installed in a virtual environment:

```
uv venv
uv sync
```

The pre-training and downstream task fine-tuning can then be called with several arguments:
```
python train.py --lr 0.0001 --num_epochs 100 --pretrain_loss mse
```

## ⚙️ Command-Line Arguments
  
This script supports the following arguments for model training, architecture setup, and data handling:

### 🔧 General Training Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--lr` | `float` | `1e-4` | Learning rate for the optimizer. |
| `--num_epochs` | `int` | `2` | Number of training epochs. |
| `--train_batch_size` | `int` | `50` | Batch size during training. |
| `--infer_batch_size` | `int` | `50` | Batch size during inference/validation. |
| `--patience` | `int` | `10` | Number of epochs to wait for early stopping. |

### 🧠 Model Architecture

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--d_model` | `int` | `64` | Embedding dimension (use `63` if `add_hour_to_embedding=True`). |
| `--dim_feedforward` | `int` | `32` | Hidden layer size in feedforward blocks. |
| `--seg_len` | `int` | `126` | Length of each input segment (e.g., frequency bins). |
| `--seq_len` | `int` | `15` | Number of sequential segments (fixed by dataset). |
| `--num_cls_token` | `int` | `1` | Number of classification tokens used. |
| `--time_ar_layer` | `int` | `2` | Number of transformer layers in the autoregressive module. |
| `--time_ar_head` | `int` | `4` | Number of attention heads. |
| `--use_rotary_encoding` | `bool` | `False` | Use rotary positional encoding in the transformer. |

### 🔄 Pretraining Settings

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--pretrain_loss` | `str` | `"mae"` | Loss function for pretraining (`"mse"` or `"mae"`). |
| `--pretrain_fooof` | `bool` | `False` | Enable FOOOF-based spectral loss. |
| `--ap_loss_factor` | `float` | `0.5` | Weight for aperiodic vs. periodic components in loss. |
| `--fooof_loss_factor` | `float` | `0.1` | Weight for FOOOF vs. spectrogram loss. |
| `--warm_up_epochs_before_fooof` | `int` | `30` | Epochs to train before applying FOOOF-based loss. |

### 📉 Downstream Task Settings

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--downstream_label` | `str` | `"all"` | Label to decode (`"pkg_bk"`, `"pkg_dk"`, `"pkg_tremor"`, `"all"`). |
| `--downstream_loss` | `str` | `"mae"` | Loss for downstream decoding (`"corr"`, `"mae"`, `"mse"`). |

### 🧪 Feature & Embedding Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--apply_log_scaling` | `bool` | `True` | Apply logarithmic scaling to the input features. |
| `--add_hour_to_embedding` | `bool` | `False` | Append hour-of-day info to input embeddings. |
| `--add_hour_to_features` | `bool` | `False` | Add hour-of-day as an extra input feature. |

### 💾 File Paths & Execution Settings

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--PATH_DATA` | `str` | `"/.../features/ts_transformer"` | Path to the dataset directory. |
| `--path_out` | `str` | `"/.../out_save_debug"` | Output directory for models and logs. |
| `--device` | `str` | `"cpu"` | Device for computation (`"cpu"` or `"cuda"`). |
| `--tb_name` | `str` | `"fm"` | TensorBoard experiment name. |
| `--sub_idx` | `int` | `6` | Subject index for debug or individual training. |
| `--load_pretrained` | `bool` | `True` | Whether to load a pretrained model. |
| `--multiprocess_on_one_machine` | `bool` | `False` | Enable multi-processing on a single node. |
