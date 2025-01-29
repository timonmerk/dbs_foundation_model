import numpy as np
from sklearn import metrics
import pandas as pd
import torch
from custom_dataset import CustomDataset
from torch.utils.data import DataLoader
from torch import nn

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def plot_predictions_downstream(writer, y_val_true: list, y_val_pred: list, classification: bool, epoch: int):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.figure()
    for idx, idx_label in enumerate(["bk", "dk", "tremor"]):
        plt.subplot(3, 1, idx+1)
        if classification:
            plt.plot(np.concatenate(y_val_true)[:, idx], label="True")
            plt.plot(sigmoid(np.concatenate(y_val_pred)[:, idx]), label="Predicted")
        else:
            plt.plot(np.concatenate(y_val_true)[:, idx], label="True")
            plt.plot(np.concatenate(y_val_pred)[:, idx], label="Predicted")
        plt.legend()
        plt.title(idx_label)
    plt.legend()
    plt.tight_layout()
    if classification:
        writer.add_figure('Classification Predictions Val', plt.gcf(), epoch)
    else:
        writer.add_figure('Regression Predictions Val', plt.gcf(), epoch)
    plt.close()

def plot_prediction_pretrain(writer, data_pred, data, mask, epoch: int):
    from matplotlib import pyplot as plt
    pred_ = data_pred.detach().cpu().numpy()
    true_ = data.detach().cpu().numpy()
    mask_ = mask.detach().cpu().numpy()

    batch_idx = 0
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(pred_[batch_idx, :, :, 0].T, aspect="auto")
    plt.title("Prediction")
    plt.gca().invert_yaxis()
    plt.clim(pred_[batch_idx, :, :, 0].min(), pred_[batch_idx, :, :, 0].max())
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(true_[batch_idx, :, :, 0].T, aspect="auto")
    plt.title("True")
    plt.colorbar()
    plt.gca().invert_yaxis()
    a_ = true_[batch_idx, :, :, 0]
    a_max = np.max(a_[np.nonzero(a_)])
    plt.clim(true_[batch_idx, :, :, 0].min(), a_max
             )
    
    writer.add_figure('Pretrain Prediction', plt.gcf(), epoch)
    plt.close()

def report_tb_res(y_pred, y_true, epoch, writer, classification=False, train_=True):
    
    if train_:
        str_prefix = "train"
    else:
        str_prefix = "val"
    
    pred_ = np.concatenate(y_pred)
    true_ = np.concatenate(y_true)

    dict_res = {}

    if classification:
        metric_use = metrics.balanced_accuracy_score
        ba_bk = metric_use(true_[:, 0], sigmoid(pred_[:, 0])>0.5)
        ba_dk = metric_use(true_[:, 1], sigmoid(pred_[:, 1])>0.5)
        ba_tremor = metric_use(true_[:, 2], sigmoid(pred_[:, 2])>0.5)

        dict_res["ba_bk"] = ba_bk
        dict_res["ba_dk"] = ba_dk
        dict_res["ba_tremor"] = ba_tremor

        writer.add_scalar(f"{str_prefix}_ba_bk", ba_bk, epoch)
        writer.add_scalar(f"{str_prefix}_ba_dk", ba_dk, epoch)
        writer.add_scalar(f"{str_prefix}_ba_tremor", ba_tremor, epoch)
    else:
        metric_use = np.corrcoef

        corr_bk = metric_use(true_[:, 0], pred_[:, 0])[0, 1]
        corr_dk = metric_use(true_[:, 1], pred_[:, 1])[0, 1]
        corr_tremor = metric_use(true_[:, 2], pred_[:, 2])[0, 1]

        dict_res["corr_bk"] = corr_bk
        dict_res["corr_dk"] = corr_dk
        dict_res["corr_tremor"] = corr_tremor

        writer.add_scalar(f"{str_prefix}_corr_bk", corr_bk, epoch)
        writer.add_scalar(f"{str_prefix}_corr_dk", corr_dk, epoch)
        writer.add_scalar(f"{str_prefix}_corr_tremor", corr_tremor, epoch)

        metric_use = metrics.mean_absolute_error
        mse_bk = metric_use(true_[:, 0], pred_[:, 0])
        mse_dk = metric_use(true_[:, 1], pred_[:, 1])
        mse_tremor = metric_use(true_[:, 2], pred_[:, 2])

        dict_res["mse_bk"] = mse_bk
        dict_res["mse_dk"] = mse_dk
        dict_res["mse_tremor"] = mse_tremor

        writer.add_scalar(f"{str_prefix}_mse_bk", mse_bk, epoch)
        writer.add_scalar(f"{str_prefix}_mse_dk", mse_dk, epoch)
        writer.add_scalar(f"{str_prefix}_mse_tremor", mse_tremor, epoch)
    
    return dict_res

def get_all_embeddings(args, encoder: nn.Module):
    subs = pd.read_csv(os.path.join(args.PATH_DATA, "all_label_series.csv"))["sub"].unique()

    all_emb = []
    labels = []
    dataset = CustomDataset(args.PATH_DATA, subs)
    data_iter_val = DataLoader(dataset, shuffle=False, batch_size=args.infer_batch_size, drop_last=False, num_workers=1)
    encoder.eval()

    with torch.no_grad():
        for idx, (data, label) in enumerate(data_iter_val):
            data = torch.tensor(data, dtype=torch.float32).to(args.device)
            label = torch.tensor(label, dtype=torch.float32).to(args.device)
            bat_size, seq_len, seg_len, ch_num = data.shape

            cls_token_emb, data_enc = encoder(data)
            data_enc = data_enc.reshape(bat_size, seq_len, ch_num, args.d_model)
            cls_token_emb = cls_token_emb.reshape(bat_size, 1, ch_num, args.d_model)
            all_emb.append(cls_token_emb.detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy())

    all_emb = np.concatenate(all_emb)
    labels = np.concatenate(labels)
    return all_emb, labels