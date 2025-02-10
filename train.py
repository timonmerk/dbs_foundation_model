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
from custom_dataset import CustomDataset
from dbs_fmodel import TimeEncoder

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train(args, encoder: nn.Module, linear: nn.Module, downstream: nn.Module,
          pretrain: bool, subs_val: list, classification: bool = False):

    writer = SummaryWriter(log_dir=args.path_summary_writer)
    if pretrain:
        if args.pretrain_loss == "mse":
            loss = nn.MSELoss()
        else:
            loss = nn.L1Loss()

        optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(linear.parameters()), lr=args.lr, betas=(0.9, 0.95))
    else:
        if args.downstream_loss == "corr":
            maximize_ = True
        else:
            maximize_ = False
        optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(downstream.parameters()), lr=args.lr, betas=(0.9, 0.95), maximize=maximize_)

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
            # data = torch.tensor(data, dtype=torch.float32).to(args.device)
            # label = torch.tensor(label, dtype=torch.float32)[:, :3].to(args.device)  # bk, dk, tremor, h
            data = data.to(torch.float32).to(args.device)
            label = label.to(torch.float32).to(args.device)[:, :3]

            bat_size, seq_len, seg_len, ch_num = data.shape

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

            cls_token_embs, data_enc = encoder(data)
            data_enc = data_enc.reshape(bat_size, seq_len, ch_num, args.d_model)
            cls_token_embs = cls_token_embs.reshape(bat_size, args.num_cls_token, ch_num, args.d_model)

            if pretrain:
                data_pred = linear(data_enc)
                data_pred = torch.transpose(data_pred, 2, 3)

                if args.pretrain_fooof and epoch >= args.warm_up_epochs_before_fooof:
                    true_fooof = utils.get_fooof_fit(data_true_mask.clone().detach().cpu().numpy())
                    pred_fooof = utils.get_fooof_fit(data_pred[:, mask, :, :].clone().detach().cpu().numpy())
                    true_fooof = torch.from_numpy(true_fooof).float().to(args.device).requires_grad_()
                    pred_fooof = torch.from_numpy(pred_fooof).float().to(args.device).requires_grad_()
                    idx_not_nan_pred = ~torch.isnan(pred_fooof).any(axis=1)
                    idx_not_nan_true = ~torch.isnan(true_fooof).any(axis=1)
                    idx_not_nan = idx_not_nan_pred & idx_not_nan_true
                    pred_fooof = pred_fooof[idx_not_nan]
                    true_fooof = true_fooof[idx_not_nan]

                    loss_ap = loss(pred_fooof[:, :2], true_fooof[:, :2])
                    loss_pe = loss(pred_fooof[:, 2:], true_fooof[:, 2:])
                    
                    loss_fooof = loss_ap * args.ap_loss_factor + loss_pe * (1 - args.ap_loss_factor)
                    loss_spec = loss(data_pred[:, mask, :, :], data_true_mask)
                    bat_loss = loss_fooof * args.fooof_loss_factor + loss_spec * (1 - args.fooof_loss_factor)
                else:
                    bat_loss = loss(data_pred[:, mask, :, :], data_true_mask)
            else:
                if classification:
                    class_counts = label.sum(axis=0)
                    if torch.sum(class_counts == 0):
                        weights = torch.ones(3)  #np.ones(3)
                    else:
                        weights = 1/(class_counts / label.shape[0])
                    loss = nn.BCEWithLogitsLoss(pos_weight=weights.to(args.device))
                else:  # downstream
                    if args.downstream_loss == "mse":
                        loss = nn.MSELoss()
                    elif args.downstream_loss == "mae":
                        loss = nn.L1Loss()
                    else:
                        loss = PearsonCorrCoef()

                cls_token_embs = cls_token_embs.reshape(bat_size, args.num_cls_token * ch_num * args.d_model)
                data_pred = torch.squeeze(downstream(cls_token_embs))
                if args.downstream_label == "all" and type(loss) == PearsonCorrCoef:
                    bat_loss_bk = loss(data_pred[:, 0], label[:, 0])
                    bat_loss_dk = loss(data_pred[:, 1], label[:, 1])
                    bat_loss_tremor = loss(data_pred[:, 2], label[:, 2])
                    
                    losses = torch.tensor([bat_loss_bk, bat_loss_dk, bat_loss_tremor]).requires_grad_(True)
                    valid_losses = losses[~torch.isnan(losses)]

                    if valid_losses.numel() > 0:  # Check if there are valid values
                        bat_loss = valid_losses.mean()
                    else:
                        bat_loss = torch.tensor(0.0).requires_grad_(True)
                else:
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
            _ = utils.report_tb_res(args, y_train_pred, y_train_true, epoch, writer, classification=classification, train_=True)

        #if epoch % 2 != 0:  # validation every 2 epochs
        #    continue

        encoder.eval()
        linear.eval()

        val_loss = 0
        y_val_pred = []
        y_val_true = []

        with torch.no_grad():
            for idx, (data, label) in enumerate(data_iter_val):
                # data = torch.tensor(data, dtype=torch.float32).to(args.device)
                # label = torch.tensor(label, dtype=torch.float32)[:, :3].to(args.device)  # bk, dk, tremor, h
                data = data.to(torch.float32).to(args.device)
                label = label.to(torch.float32).to(args.device)[:, :3]

                bat_size, seq_len, seg_len, ch_num = data.shape

                if args.downstream_label == "pkg_bk":
                    label = label[:, 0]
                elif args.downstream_label == "pkg_dk":
                    label = label[:, 1]
                elif args.downstream_label == "pkg_tremor":
                    label = label[:, 2]

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

                    if args.pretrain_fooof and epoch >= args.warm_up_epochs_before_fooof:
                        true_fooof = utils.get_fooof_fit(data_true_mask.detach().cpu().numpy())
                        pred_fooof = utils.get_fooof_fit(data_pred[:, mask, :, :].detach().cpu().numpy())
                        true_fooof = torch.tensor(true_fooof, dtype=torch.float32).to(args.device)
                        pred_fooof = torch.tensor(pred_fooof, dtype=torch.float32).to(args.device)
                        idx_not_nan_pred = ~torch.isnan(pred_fooof).any(axis=1)
                        idx_not_nan_true = ~torch.isnan(true_fooof).any(axis=1)
                        idx_not_nan = idx_not_nan_pred & idx_not_nan_true
                        pred_fooof = pred_fooof[idx_not_nan]
                        true_fooof = true_fooof[idx_not_nan]
                        loss_ap = loss(pred_fooof[:, :2], true_fooof[:, :2])
                        loss_pe = loss(pred_fooof[:, 2:], true_fooof[:, 2:])

                        loss_fooof = loss_ap * args.ap_loss_factor + loss_pe * (1 - args.ap_loss_factor)
                        loss_spec = loss(data_pred[:, mask, :, :], data_true_mask)
                        bat_loss = loss_fooof * args.fooof_loss_factor + loss_spec * (1 - args.fooof_loss_factor)

                    else:
                        bat_loss = loss(data_pred[:, mask, :, :], data_true_mask)

                else:  # downstream
                    cls_token_embs = cls_token_embs.reshape(bat_size, args.num_cls_token * ch_num * args.d_model)
                    data_pred = torch.squeeze(downstream(cls_token_embs))

                    if classification:
                        class_counts = label.sum(axis=0)
                        if torch.sum(class_counts == 0):
                            weights = torch.ones(3)
                        else:
                            weights = 1/(class_counts / label.shape[0])
                        loss = nn.BCEWithLogitsLoss(pos_weight=weights.to(args.device))
                    else:
                        if args.downstream_loss == "mse":
                            loss = nn.MSELoss()
                        elif args.downstream_loss == "mae":
                            loss = nn.L1Loss()
                        else:
                            loss = PearsonCorrCoef()
    
                    if args.downstream_label == "all" and type(loss) == PearsonCorrCoef:
                        bat_loss_bk = loss(data_pred[:, 0], label[:, 0])
                        bat_loss_dk = loss(data_pred[:, 1], label[:, 1])
                        bat_loss_tremor = loss(data_pred[:, 2], label[:, 2])

                        losses = torch.tensor([bat_loss_bk, bat_loss_dk, bat_loss_tremor])
                        valid_losses = losses[~torch.isnan(losses)]

                        if valid_losses.numel() > 0:  # Check if there are valid values
                            bat_loss = valid_losses.mean()
                        else:
                            bat_loss = torch.tensor(0.0)
                    else:
                        bat_loss = loss(data_pred, label)
                    y_val_pred.append(data_pred.detach().cpu().numpy())
                    y_val_true.append(label.detach().cpu().numpy())

                val_loss += bat_loss.item()

            if not pretrain:
                dict_res = utils.report_tb_res(args, y_val_pred, y_val_true, epoch, writer, classification=classification, train_=False)

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
                utils.plot_prediction_pretrain(writer, data_pred, data, mask, epoch)

            else:
                torch.save(encoder.state_dict(), os.path.join(args.path_downstream, 'finetuned_encoder.pth'))
                torch.save(downstream.state_dict(), os.path.join(args.path_downstream, 'finetuned_downstream.pth'))
            
                utils.plot_predictions_downstream(writer, args, y_val_true, y_val_pred, classification, epoch)

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

    if args.downstream_label == "all":
        out_dim = 3
    else:
        out_dim = 1

    downstream = nn.Sequential(
        nn.Linear(in_features=args.d_model * 2 * args.num_cls_token, out_features=32),
        nn.ReLU(),
        nn.Linear(in_features=32, out_features=out_dim)
    ).to(args.device)

    args.path_pretrain = os.path.join(args.path_pretrain_base, sub_val)
    args.path_downstream = os.path.join(args.path_downstream_base, sub_val)
    
    args.tb_name = args.tb_name + f"_{sub_val}"
    args.path_summary_writer = os.path.join(args.path_out, "runs", args.tb_name)
    os.makedirs(args.path_summary_writer, exist_ok=True)

    os.system(f"rm -r {args.path_summary_writer}")

    os.makedirs(args.path_pretrain, exist_ok=True)

    train(args, encoder, linear, downstream=downstream, pretrain=True, subs_val=sub_hem_val_list)

    for classification in [False, True]:
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
    parser.add_argument("--num_epochs", type=int, default=100)  # 20
    parser.add_argument("--train_batch_size", type=int, default=50)
    parser.add_argument("--infer_batch_size", type=int, default=50)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--dim_feedforward", type=int, default=8)  # 32
    parser.add_argument("--seg_len", type=int, default=126)  # frequency bins
    parser.add_argument("--seq_len", type=int, default=15)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--pretrain_loss", type=str, default="mae", choices=["mse", "mae"])
    parser.add_argument("--downstream_label", type=str, default="pkg_tremor", choices=["pkg_bk", "pkg_dk", "pkg_tremor", "all"])
    parser.add_argument("--downstream_loss", type=str, default="mae", choices=["corr", "mae", "mse"])
    parser.add_argument("--pretrain_fooof", type=bool, default=True)
    parser.add_argument("--ap_loss_factor", type=float, default=0.5)  # how much periodic and aperiodic losses are weighted
    parser.add_argument("--fooof_loss_factor", type=float, default=0.1)  # how much fooof and spectrogram losses are weighted
    parser.add_argument("--warm_up_epochs_before_fooof", type=int, default=30)
    parser.add_argument("--load_pretrained", type=bool, default=True)
    parser.add_argument("--use_rotary_encoding", type=bool, default=False)
    parser.add_argument("--num_cls_token", type=int, default=1)
    parser.add_argument("--time_ar_layer", type=int, default=2)  # 4
    parser.add_argument("--time_ar_head", type=int, default=4)  # 8
    parser.add_argument("--PATH_DATA", type=str, default="/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/ts_transformer")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--path_out", type=str, default="/Users/Timon/Documents/dbs_foundation_model/out_save")
    parser.add_argument("--tb_name", type=str, default='fm')
    parser.add_argument("--sub_idx", type=int, default=0)
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
        max_parallel_processes = 5

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

        save_res_combined(sub_unique, args)
