import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics.regression import PearsonCorrCoef
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
import utils
from queue import Queue
from custom_dataset import CustomDataset
from dbs_fmodel import TimeEncoder

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train(args, encoder: nn.Module, linear: nn.Module, downstream: nn.Module,
          pretrain: bool, subs_val: list, classification: bool = False):

    writer = SummaryWriter(log_dir=args.path_summary_writer)

    if args.pretrain_loss == "mse":
        loss = nn.MSELoss()
    else:
        loss = nn.L1Loss()

    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(linear.parameters()), lr=args.lr, betas=(0.9, 0.95))

    subs = pd.read_csv(os.path.join(args.PATH_DATA, "all_label_series.csv"))["sub"].unique()
    subs_train = [sub for sub in subs]  # if sub not in subs_val

    dataset_train = CustomDataset(args.PATH_DATA, subs_train, classification=classification)

    data_iter_train = DataLoader(dataset_train, shuffle=True, batch_size=args.train_batch_size, drop_last=False, num_workers=1, pin_memory=False, persistent_workers=True)
    
    encoder.train()
    linear.train()

    epochs_no_improve = 0
    best_val_loss = float('inf')
    early_stop = False

    y_train_pred = []
    y_train_true = []
    d_res_all = {}
    embs_ = []
    hours_ = []
    labels_ = []
    for epoch in range(args.num_epochs):

        train_loss = 0
        for idx, (data, label) in enumerate(data_iter_train):
            # data = torch.tensor(data, dtype=torch.float32).to(args.device)
            # label = torch.tensor(label, dtype=torch.float32)[:, :3].to(args.device)  # bk, dk, tremor, h
            
            bat_size, seq_len, seg_len, ch_num = data.shape

            if args.add_hour_to_features:
                hour = label[:, 3]
                hour_expanded = hour[:, None, None, None].expand(bat_size, seq_len, 1, ch_num)  # hour is 1d
                data = torch.cat((data, hour_expanded), dim=2)
 
            data = data.to(torch.float32).to(args.device)
            hour = label[:, 3].to(torch.float32).to(args.device)
            label = label.to(torch.float32).to(args.device)[:, :2]

            if args.downstream_label == "pkg_bk":
                label = label[:, 0]
            elif args.downstream_label == "pkg_dk":
                label = label[:, 1]
            elif args.downstream_label == "pkg_tremor":
                label = label[:, 2]
            if pretrain:
                # set randomly 3 out of 15 time points to zero
                mask = torch.rand(data.shape[1]) < 0.3
                
                while mask.sum() == 0:
                    mask = torch.rand(data.shape[1]) < 0.3
                    if mask.sum() > 0:
                        break
                data_true_mask = data[:, mask, :, :].clone()
                data[:, mask, :, :] = 0

            cls_token_embs, data_enc = encoder(data, hour)
            d_model_ = args.d_model + 1 if args.add_hour_to_embedding else args.d_model
            data_enc = data_enc.reshape(bat_size, seq_len, ch_num, d_model_)
            cls_token_embs = cls_token_embs.reshape(bat_size, args.num_cls_token, ch_num, d_model_)

            if epoch == args.num_epochs - 1:
                embs_.append(cls_token_embs.detach().cpu().numpy())
                labels_.append(label.detach().cpu().numpy())
                hours_.append(hour.detach().cpu().numpy())

            data_pred = linear(data_enc)
            data_pred = torch.transpose(data_pred, 2, 3)


            bat_loss = loss(data_pred[:, mask, :, :], data_true_mask)
            train_loss += bat_loss.item()

            optimizer.zero_grad()
            bat_loss.backward()
            optimizer.step()
        loss_name = "train_pretrain_loss"

        writer.add_scalar(loss_name, train_loss, epoch)
        logging.info(f"Epoch: {epoch}, Train Loss: {train_loss}")



    # Save the results
    if pretrain:
        with open(os.path.join(args.path_downstream, 'dict_res_emb.pkl'), 'wb') as f:
            emb_dict = {"embs": embs_, "labels": labels_, "hours": hours_}
            pickle.dump(emb_dict, f)
    
def cv_runner(args, sub_idx_test):

    sub_hem_unique = pd.read_csv(os.path.join(args.PATH_DATA, "all_label_series.csv"))["sub"].unique()
    sub_unique = np.unique([sub[:-1] for sub in sub_hem_unique])
    sub_val = sub_unique[sub_idx_test]
    
    sub_hem_val_list = [hem for hem in sub_hem_unique if hem.startswith(sub_val)]
    
    in_dim_ = args.seg_len +1  if args.add_hour_to_features else args.seg_len
    encoder = TimeEncoder(in_dim=in_dim_,
                            d_model=args.d_model,
                            dim_feedforward=args.dim_feedforward,
                            seq_len=args.seq_len,
                            n_layer=args.time_ar_layer,
                            nhead=args.time_ar_head,
                            num_cls_token=args.num_cls_token,
                            use_rotary_encoding=args.use_rotary_encoding,
                            add_hour_emb=args.add_hour_to_embedding
                            ).to(args.device)
    
    logging.info(f"Number of parameters: {sum(p.numel() for p in encoder.parameters())}")
    
    d_model_ = args.d_model + 1 if args.add_hour_to_embedding else args.d_model
    linear = nn.Linear(in_features=d_model_, out_features=in_dim_).to(args.device)

    if args.downstream_label == "all":
        out_dim = 2
    else:
        out_dim = 1

    downstream = nn.Sequential(
        nn.Linear(in_features=d_model_ * 2 * args.num_cls_token, out_features=32),
        nn.ReLU(),
        nn.Linear(in_features=32, out_features=out_dim)
    ).to(args.device)

    args.path_pretrain = os.path.join(args.path_pretrain_base, sub_val)
    args.path_downstream = args.path_downstream_base
    
    args.tb_name = args.tb_name + f"_{sub_val}"
    args.path_summary_writer = os.path.join(args.path_out, "runs", args.tb_name)
    os.makedirs(args.path_summary_writer, exist_ok=True)

    os.system(f"rm -r {args.path_summary_writer}")

    os.makedirs(args.path_pretrain, exist_ok=True)

    train(args, encoder, linear, downstream=downstream, pretrain=True, subs_val=sub_hem_val_list)


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
    parser.add_argument("--num_epochs", type=int, default=100)  # 100
    parser.add_argument("--train_batch_size", type=int, default=50)
    parser.add_argument("--infer_batch_size", type=int, default=50)
    parser.add_argument("--d_model", type=int, default=64)  # 63 if add_hour_to_embedding
    parser.add_argument("--dim_feedforward", type=int, default=32)  # 32
    parser.add_argument("--seg_len", type=int, default=126)  # frequency bins
    parser.add_argument("--seq_len", type=int, default=15)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--pretrain_loss", type=str, default="mae", choices=["mse", "mae"])
    parser.add_argument("--downstream_label", type=str, default="all", choices=["pkg_bk", "pkg_dk", "pkg_tremor", "all"])  # all is currently just bk and dk
    parser.add_argument("--downstream_loss", type=str, default="mae", choices=["corr", "mae", "mse"])
    parser.add_argument("--pretrain_fooof", type=bool, default=False)
    parser.add_argument("--ap_loss_factor", type=float, default=0.5)  # how much periodic and aperiodic losses are weighted
    parser.add_argument("--fooof_loss_factor", type=float, default=0.1)  # how much fooof and spectrogram losses are weighted
    parser.add_argument("--warm_up_epochs_before_fooof", type=int, default=30)
    parser.add_argument("--add_hour_to_embedding", type=bool, default=False)
    parser.add_argument("--add_hour_to_features", type=bool, default=True)
    parser.add_argument("--load_pretrained", type=bool, default=True)
    parser.add_argument("--use_rotary_encoding", type=bool, default=False)
    parser.add_argument("--num_cls_token", type=int, default=1)
    parser.add_argument("--time_ar_layer", type=int, default=2)  # 4
    parser.add_argument("--time_ar_head", type=int, default=4)  # 8
    parser.add_argument("--PATH_DATA", type=str, default="/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/ts_transformer")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--path_out", type=str, default="/Users/Timon/Documents/dbs_foundation_model/out_save_debug")
    parser.add_argument("--tb_name", type=str, default='fm')
    parser.add_argument("--sub_idx", type=int, default=6)  # 0
    parser.add_argument("--multiprocess_on_one_machine", type=bool, default=False)

    args = parser.parse_args()

    args.path_pretrain_base = os.path.join(args.path_out, "models_save")
    args.path_downstream_base = os.path.join(args.path_out, "models_save_downstream")
    
    sub_hem_unique = pd.read_csv(os.path.join(args.PATH_DATA, "all_label_series.csv"))["sub"].unique()
    sub_unique = np.unique([sub[:-1] for sub in sub_hem_unique])

    mp.set_start_method('spawn', force=True)

    if not args.multiprocess_on_one_machine:
        process_wrapper(args, args.sub_idx)
    else:
        max_parallel_processes = sub_unique.shape[0]

        processes = []
        for sub_idx_test in range(sub_unique.shape[0]):
            p = mp.Process(target=process_wrapper, args=(args, sub_idx_test,))
            p.start()
            processes.append(p)

            # Check if we reached the maximum number of parallel processes
            if len(processes) >= max_parallel_processes:
                # Wait for the first process to finish
                processes[0].join()
                # Remove the finished process from the list
                processes.pop(0)

        for p in processes:
            p.join()

        #save_res_combined(sub_unique, args)
