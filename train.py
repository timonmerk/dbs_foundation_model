import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import math
import argparse
import os
from tqdm import tqdm
from sklearn import metrics
import pickle
from torch.utils.tensorboard import SummaryWriter
from joblib import Parallel, delayed
import torch.multiprocessing as mp

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

        metric_use = metrics.mean_squared_error
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
        
class CustomDataset(Dataset):

    def __init__(self, PATH_DATA, train_subs: list, classification=False):

        self.data = np.load(os.path.join(PATH_DATA, "all_arr_series.npy"))
        self.labels = pd.read_csv(os.path.join(PATH_DATA, "all_label_series.csv"))
        self.labels["pkg_dt"] = pd.to_datetime(self.labels["pkg_dt"])
        self.labels["pkg_h"] = self.labels["pkg_dt"].dt.hour
        self.labels["sub_id"] = pd.Categorical(self.labels["sub"]).codes

        self.labels["pkg_bk_class"] = self.labels["pkg_bk"] > 50
        self.labels["pkg_tremor_class"] = self.labels["pkg_tremor"] > 0
        # for each patient min max normalize the pkg_dk and set the threshold to 0.02
        self.labels["pkg_dk_class"] = self.labels.groupby("sub")["pkg_dk"].transform(lambda x: (x - x.min()) / (x.max() - x.min())) > 0.02

        idx_subs = self.labels["sub"].isin(train_subs)
    
        if classification:
            self.labels = self.labels[idx_subs].reset_index(drop=True)[[
                "pkg_bk_class", "pkg_dk_class", "pkg_tremor_class", "pkg_h", "sub_id"]
            ].values.astype(np.float32)
        else:
            self.labels = self.labels[idx_subs].reset_index(drop=True)[[
                "pkg_bk", "pkg_dk", "pkg_tremor", "pkg_h", "sub_id"]
            ].values.astype(np.float32)

        self.data = self.data[idx_subs]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim):
        super(RotaryPositionalEncoding, self).__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, t):
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return emb

class InputEmbedding(nn.Module):
    def __init__(self, in_dim, seq_len, d_model, num_cls_token, use_rotary_encoding=True):
        super(InputEmbedding, self).__init__()

        self.cls_tokens = nn.Parameter(torch.randn(num_cls_token, 1, d_model), requires_grad=True)  # classification tokens
        if use_rotary_encoding:
            self.positional_encoding = RotaryPositionalEncoding(d_model)
        else:
            self.positional_encoding = nn.Parameter(torch.randn(seq_len + num_cls_token, d_model), requires_grad=True)  # learnable positional encoding
        self.use_rotary_encoding = use_rotary_encoding
        self.num_cls_token = num_cls_token
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
        )
        self.apply(_weights_init)

    def forward(self, data):
        bat_size, seq_len, seg_len, ch_num = data.shape
        data = data.view(bat_size * ch_num, seq_len, seg_len)
        input_emb = self.proj(data)
        
        cls_tokens = self.cls_tokens.expand(-1, bat_size * ch_num, -1).transpose(0, 1)
        input_emb = torch.cat((cls_tokens, input_emb), dim=1)
        
        if self.use_rotary_encoding:
            pos_enc = self.positional_encoding(torch.arange(seq_len + self.num_cls_token, device=input_emb.device).float())
        else:
            pos_enc = self.positional_encoding
        input_emb += pos_enc

        return input_emb

def _weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class TimeEncoder(nn.Module):
    def __init__(self, in_dim, d_model, dim_feedforward, seq_len, n_layer, nhead, num_cls_token, use_rotary_encoding=True):
        super(TimeEncoder, self).__init__()

        self.input_embedding = InputEmbedding(in_dim=in_dim, seq_len=seq_len, d_model=d_model, num_cls_token=num_cls_token, use_rotary_encoding=use_rotary_encoding)
        self.num_cls_token = num_cls_token

        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.trans_enc = nn.TransformerEncoder(enc_layer, num_layers=n_layer)

        self.apply(_weights_init)

    def forward(self, data):
        input_emb = self.input_embedding(data)
        trans_out = self.trans_enc(input_emb)
        cls_token_outputs = trans_out[:, :self.num_cls_token, :]  # extract the classification token outputs

        return cls_token_outputs, trans_out[:, self.num_cls_token:, :]

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

def train(args, encoder: nn.Module, linear: nn.Module, downstream: nn.Module,
          pretrain: bool, subs_val: list, classification: bool = False):

    writer = SummaryWriter(log_dir=args.path_summary_writer)
    if pretrain:
        loss = nn.MSELoss()
        optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(linear.parameters()), lr=args.lr, betas=(0.9, 0.95))
    else:
        optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(downstream.parameters()), lr=args.lr, betas=(0.9, 0.95))
        if classification: 
            loss = nn.BCEWithLogitsLoss()
        else:
            loss = nn.MSELoss()

    subs = pd.read_csv(os.path.join(args.PATH_DATA, "all_label_series.csv"))["sub"].unique()
    subs_train = [sub for sub in subs if sub not in subs_val]

    dataset_train = CustomDataset(args.PATH_DATA, subs_train, classification=classification)
    dataset_val = CustomDataset(args.PATH_DATA, subs_val, classification=classification)

    data_iter_train = DataLoader(dataset_train, shuffle=True, batch_size=args.train_batch_size, drop_last=False, num_workers=1)
    data_iter_val = DataLoader(dataset_val, shuffle=False, batch_size=args.infer_batch_size, drop_last=False, num_workers=1)
    
    encoder.train()
    linear.train()

    epochs_no_improve = 0
    best_val_loss = float('inf')
    early_stop = False

    y_train_pred = []
    y_train_true = []
    
    for epoch in range(args.num_epochs):

        train_loss = 0

        for idx, (data, label) in enumerate(data_iter_train):
            data = torch.tensor(data, dtype=torch.float32).to(args.device)
            bat_size, seq_len, seg_len, ch_num = data.shape
            label = torch.tensor(label, dtype=torch.float32)[:, :3].to(args.device)  # bk, dk, tremor, h

            if pretrain:
                # set randomly 3 out of 15 time points to zero
                mask = torch.rand(data.shape[1]) < 0.3
                
                while mask.sum() == 0:
                    mask = torch.rand(data.shape[1]) < 0.3
                    if mask.sum() > 0:
                        break
                data_true_mask = data[:, mask, :, :].clone()
                data[:, mask, :, :] = 0

            cls_token_embs, data_enc = encoder(data)
            data_enc = data_enc.reshape(bat_size, seq_len, ch_num, args.d_model)
            cls_token_embs = cls_token_embs.reshape(bat_size, args.num_cls_token, ch_num, args.d_model)

            if pretrain:
                data_pred = linear(data_enc)
                data_pred = torch.transpose(data_pred, 2, 3)
                bat_loss = loss(data_pred[:, mask, :, :], data_true_mask)
            else:
                if classification:
                    class_counts = label.sum(axis=0)
                    if torch.sum(class_counts == 0):
                        weights = torch.ones(3)  #np.ones(3)
                    else:
                        weights = 1/(class_counts / label.shape[0])
                    loss = nn.BCEWithLogitsLoss(pos_weight=weights.to(args.device))
                else:
                    loss = nn.MSELoss()
            
                cls_token_embs = cls_token_embs.reshape(bat_size, args.num_cls_token * ch_num * args.d_model)
                data_pred = torch.squeeze(downstream(cls_token_embs))
                bat_loss = loss(data_pred, label)

                y_train_pred.append(data_pred.detach().cpu().numpy())
                y_train_true.append(label.detach().cpu().numpy())

            train_loss += bat_loss.item()

            optimizer.zero_grad()
            bat_loss.backward()
            optimizer.step()
            logging.info(f'Epoch [{epoch+1}/{args.num_epochs}], Batch [{idx+1}/{len(data_iter_train)}], Training Loss: {bat_loss:.4f}')
        
        if pretrain:
            loss_name = "train_pretrain_loss"
        else:
            if classification:
                loss_name = "train_finetune_loss_classification"
            else:
                loss_name = "train_finetune_loss_regression"
        writer.add_scalar(loss_name, train_loss, epoch)
        logging.info(f"Epoch: {epoch}, Train Loss: {train_loss}")

        if not pretrain:
            _ = report_tb_res(y_train_pred, y_train_true, epoch, writer, classification=classification, train_=True)

        #if epoch % 2 != 0:  # validation every 2 epochs
        #    continue

        encoder.eval()
        linear.eval()

        val_loss = 0
        y_val_pred = []
        y_val_true = []

        with torch.no_grad():
            for idx, (data, label) in enumerate(data_iter_val):
                data = torch.tensor(data, dtype=torch.float32).to(args.device)
                label = torch.tensor(label, dtype=torch.float32)[:, :3].to(args.device)
                bat_size, seq_len, seg_len, ch_num = data.shape

                if pretrain:
                    # set randomly 3 out of 15 time points to zero
                    mask = torch.rand(data.shape[1]) < 0.3
                    
                    while mask.sum() == 0:
                        mask = torch.rand(data.shape[1]) < 0.3
                        if mask.sum() > 0:
                            break
                    data_true_mask = data[:, mask, :, :].clone()
                    data[:, mask, :, :] = 0

                cls_token_embs, data_enc = encoder(data)
                data_enc = data_enc.reshape(bat_size, seq_len, ch_num, args.d_model)
                cls_token_embs = cls_token_embs.reshape(bat_size, args.num_cls_token, ch_num, args.d_model)

                if pretrain: 
                    data_pred = linear(data_enc)
                    data_pred = torch.transpose(data_pred, 2, 3)
                    bat_loss = loss(data_pred[:, mask, :, :], data_true_mask)
                else:
                    cls_token_embs = cls_token_embs.reshape(bat_size, args.num_cls_token * ch_num * args.d_model)
                    data_pred = downstream(cls_token_embs)

                    if classification:
                        class_counts = label.sum(axis=0)
                        if torch.sum(class_counts == 0):
                            weights = torch.ones(3)
                        else:
                            weights = 1/(class_counts / label.shape[0])
                        loss = nn.BCEWithLogitsLoss(pos_weight=weights.to(args.device))
                    else:
                        loss = nn.MSELoss()
    
                    bat_loss = loss(data_pred, label)
                    y_val_pred.append(data_pred.detach().cpu().numpy())
                    y_val_true.append(label.detach().cpu().numpy())

                val_loss += bat_loss.item()

            if not pretrain:
                dict_res = report_tb_res(y_val_pred, y_val_true, epoch, writer, classification=classification, train_=False)

        if pretrain:
            loss_name = "val_pretrain_loss"
        else:
            if classification:
                loss_name = "val_finetune_loss_classification"
            else:
                loss_name = "val_finetune_loss_regression"
        writer.add_scalar(loss_name, val_loss, epoch)
        logging.info(f"Epoch: {epoch}, Val Loss: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            if pretrain:
                torch.save(encoder.state_dict(), os.path.join(args.path_pretrain, 'pretrain_encoder.pth'))
                torch.save(linear.state_dict(), os.path.join(args.path_pretrain, 'pretrain_linear.pth'))
                plot_prediction_pretrain(writer, data_pred, data, mask, epoch)

            else:
                torch.save(encoder.state_dict(), os.path.join(args.path_downstream, 'finetuned_encoder.pth'))
                torch.save(downstream.state_dict(), os.path.join(args.path_downstream, 'finetuned_downstream.pth'))
            
                plot_predictions_downstream(writer, y_val_true, y_val_pred, classification, epoch)

                with open(os.path.join(args.path_downstream, 'dict_res.pkl'), 'wb') as f:
                    pickle.dump(dict_res, f)

            logging.info(f"Saved best model at epoch {epoch} with validation loss {val_loss}")
        else:
            epochs_no_improve += 1

        if not pretrain and epochs_no_improve >= args.patience:
            logging.info(f"Early stopping at epoch {epoch} due to no improvement in validation loss for {args.patience} epochs")
            early_stop = True
            break

        encoder.train()
        linear.train()

    if early_stop:
        logging.info("Early stopping triggered")

def cv_runner(args, sub_idx_test):

    sub_hem_unique = pd.read_csv(os.path.join(args.PATH_DATA, "all_label_series.csv"))["sub"].unique()
    sub_unique = np.unique([sub[:-1] for sub in sub_hem_unique])
    sub_val = sub_unique[sub_idx_test]
    
    sub_hem_val_list = [hem for hem in sub_hem_unique if hem.startswith(sub_val)]
    
    in_dim_ = args.seg_len
    encoder = TimeEncoder(in_dim=in_dim_,
                            d_model=args.d_model,
                            dim_feedforward=args.dim_feedforward,
                            seq_len=args.seq_len,
                            n_layer=args.time_ar_layer,
                            nhead=args.time_ar_head,
                            num_cls_token=args.num_cls_token,
                            use_rotary_encoding=args.use_rotary_encoding).to(args.device)
    
    logging.info(f"Number of parameters: {sum(p.numel() for p in encoder.parameters())}")
    
    linear = nn.Linear(in_features=args.d_model, out_features=in_dim_).to(args.device)
    downstream = nn.Sequential(
        nn.Linear(in_features=args.d_model * 2 * args.num_cls_token, out_features=32),
        nn.ReLU(),
        nn.Linear(in_features=32, out_features=3)
    ).to(args.device)

    args.path_pretrain = os.path.join(args.path_pretrain_base, sub_val)
    args.path_downstream = os.path.join(args.path_downstream_base, sub_val)
    
    args.tb_name = args.tb_name + f"_{sub_val}"
    args.path_summary_writer = os.path.join(args.path_out, "runs", args.tb_name)
    os.makedirs(args.path_summary_writer, exist_ok=True)

    os.system(f"rm -r {args.path_summary_writer}")

    os.makedirs(args.path_pretrain, exist_ok=True)

    train(args, encoder, linear, downstream=downstream, pretrain=True, subs_val=sub_hem_val_list)

    for classification in [True, False]:
        if classification:
            pretext = "classification"
        else:
            pretext = "regression"
        
        args.path_downstream = os.path.join(args.path_downstream_base, pretext, sub_val)
        os.makedirs(args.path_downstream, exist_ok=True)

        if args.load_pretrained:
            encoder.load_state_dict(torch.load(os.path.join(args.path_pretrain, 'pretrain_encoder.pth')))
            linear.load_state_dict(torch.load(os.path.join(args.path_pretrain, 'pretrain_linear.pth')))
            logging.info("Loaded pretrained model")
        train(args, encoder, linear, downstream=downstream, pretrain=False, subs_val=sub_hem_val_list, classification=classification)

def process_wrapper(args, sub_idx_test):
    return cv_runner(args, sub_idx_test)

def save_res_combined(sub_unique, args):
    dict_res = []
    for ml_task in ["regression", "classification"]:
        for sub_idx_test in range(sub_unique.shape[0]):
            sub_val = sub_unique[sub_idx_test]
            with open(os.path.join(args.path_downstream_base, ml_task, sub_val, 'dict_res.pkl'), 'rb') as f:
                res_sub = pickle.load(f)
                res_sub["sub"] = sub_val
            dict_res.append(res_sub)
    df = pd.DataFrame(dict_res)
    df.set_index('sub', inplace=True)
    df_merged = df.groupby(df.index).first()
    df_merged.reset_index(inplace=True)
    df_merged.to_csv(os.path.join(args.path_out, "results.csv"), index=False)

if __name__ == "__main__":

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    random.seed(1)
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=30)  # 20
    parser.add_argument("--train_batch_size", type=int, default=50)
    parser.add_argument("--infer_batch_size", type=int, default=50)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--dim_feedforward", type=int, default=32)
    parser.add_argument("--seg_len", type=int, default=126)  # frequency bins
    parser.add_argument("--seq_len", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--load_pretrained", type=bool, default=True)
    parser.add_argument("--use_rotary_encoding", type=bool, default=False)
    parser.add_argument("--num_cls_token", type=int, default=1)
    parser.add_argument("--time_ar_layer", type=int, default=4)
    parser.add_argument("--time_ar_head", type=int, default=8)
    parser.add_argument("--PATH_DATA", type=str, default="/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/ts_transformer")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--path_out", type=str, default="/Users/Timon/Documents/dbs_foundation_model")
    parser.add_argument("--tb_name", type=str, default='fm')
    args = parser.parse_args()

    args.path_pretrain_base = os.path.join(args.path_out, "models_save")
    args.path_downstream_base = os.path.join(args.path_out, "models_save_downstream")
    

    sub_hem_unique = pd.read_csv(os.path.join(args.PATH_DATA, "all_label_series.csv"))["sub"].unique()
    sub_unique = np.unique([sub[:-1] for sub in sub_hem_unique])

    mp.set_start_method('spawn', force=True)

    #process_wrapper(args, 0)

    processes = []
    for sub_idx_test in range(sub_unique.shape[0]):
        p = mp.Process(target=process_wrapper, args=(args, sub_idx_test,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    save_res_combined(sub_unique, args)