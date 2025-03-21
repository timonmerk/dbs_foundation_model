import numpy as np
from sklearn import metrics
import pandas as pd
from fooof import FOOOFGroup
import torch
from custom_dataset import CustomDataset
from torch.utils.data import DataLoader
from torch import nn
import os


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_fooof_fit(data):

    # from matplotlib import pyplot as plt

    # # check input spectra
    # for idx_ in range(20):
    #     plt.imshow(data[idx_, :, :, 0].T, aspect="auto")
    #     plt.gca().invert_yaxis()
    #     a_ = data[idx_, :, :, 0]
    #     a_max = np.max(a_[np.nonzero(a_)])
    #     plt.clim(data[idx_, :, :, 0].min(), a_max)
    #     plt.show(block=True)

    # for idx_ in range(20):
    #     plt.plot(data[idx_, 0, :, 0])
    # plt.show(block=True)

    # # reshape data and keep only second dim
    # data_r = np.reshape(data, shape=(-1, data.shape[2]), order="C")

    # data_myr = []
    # for idx_b in range(data.shape[0]):
    #     for idx_s in range(data.shape[1]):
    #         for ch_idx in range(2):
    #             data_myr.append(data[idx_b, idx_s, :, ch_idx])
    # data_myr = np.array(data_myr)

    data_myr = np.reshape(np.transpose(data, (0, 1, 3, 2)), (-1, data.shape[2]))

    # plt.imshow(data_myr.T, aspect="auto")
    # plt.gca().invert_yaxis()
    # plt.show(block=True)


    # for idx_ in range(20):
    #     plt.plot(data_myr[idx_, :])
    # plt.show(block=True)

    # data_ = data.reshape(-1, data.shape[2])
    data_ = data_myr.copy()

    fg = FOOOFGroup()
    fg.verbose = False
    freqs = np.arange(0, 126, 1)
    
    fg.fit(freqs, 10**(data_[:, :]), freq_range=[2, 110])

    offsets = fg.get_params("aperiodic_params", "offset")
    exponents = fg.get_params("aperiodic_params", "exponent")

    ap_spec = offsets[:, np.newaxis] - np.log10(freqs[np.newaxis, 1:] ** exponents[:, np.newaxis])
    spec_wo_ac = data_[:, 1:] - ap_spec

    res_ = []
    res_.append(offsets)
    res_.append(exponents)
    bands_ = [[1, 3], [4, 8], [8, 12], [13, 20], [20, 35], [36, 58], [62, 80], [81, 124]]
    for band in bands_:
        idx_band = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
        res_.append(np.sum(spec_wo_ac[:, idx_band], axis=1))

    DEBUG_ = False
    if DEBUG_:
        fg.plot()
        from matplotlib import pyplot as plt
        for idx_ in range(20):
            plt.figure()
            plt.plot(freqs[1:], data_[idx_, 1:], label="Original")
            plt.plot(freqs[1:], ap_spec[idx_, :], label="Aperiodic")
            #plt.plot(freqs[1:], spec_wo_ac[0, :], label="Without Aperiodic")
            plt.show(block=True)

        plt.figure()
        plt.subplot(121)
        plt.imshow(data_.T, aspect="auto")
        plt.title("Original")
        # flip y axis
        plt.gca().invert_yaxis()
        plt.subplot(122)
        plt.imshow(ap_spec.T, aspect="auto")
        plt.gca().invert_yaxis()
        plt.title("Aperiodic")
        plt.show(block=True)

        for ap_idx in range(50):
            #plt.plot(ap_spec[ap_idx, :])
            plt.plot(spec_wo_ac[ap_idx, :], color="black", alpha=0.01)
        plt.show(block=True)

        plt.hist(exponents, bins=50)
        plt.show(block=True)



        idx_exp_neg = np.where(exponents < 0)[0]
        idx_pos = np.where(exponents > 0)[0]
        if idx_exp_neg.shape[0] > 0:
            plt.subplot(221)
            for idx_ in idx_exp_neg:
                plt.plot(data_[idx_, :], alpha=0.2, color="black")
            plt.subplot(222)
            plt.imshow(data_[idx_exp_neg].T, aspect="auto")
            plt.gca().invert_yaxis()
            plt.subplot(223)
            for idx_ in idx_pos:
                plt.plot(data_[idx_, :],alpha=0.005, color="black")
        plt.subplot(224)
        plt.imshow(data_[idx_pos].T, aspect="auto")
        plt.gca().invert_yaxis()
        plt.show(block=True)

        res_r = np.array(res_).T
        # offset, exp, ba
        for idx_plt in range(res_r.shape[1]):
            plt.subplot(5, 2, idx_plt+1)
            plt.hist(res_r[:, idx_plt], bins=50)
            if idx_plt == 0:
                plt.title("offset")
            elif idx_plt == 1:
                plt.title("exp")
            else:
                plt.title(f"{bands_[idx_plt-2]}")
        plt.tight_layout()
        plt.show(block=True)
    return np.array(res_).T

def plot_predictions_downstream(writer, args, y_val_true: list, y_val_pred: list, classification: bool, epoch: int):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.figure()
    if args.downstream_label == "all":
        for idx, idx_label in enumerate(["bk", "dk"]):  # "tremor"
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
    else:
        plt.subplot(1, 1, 1)
        if classification:
            plt.plot(np.concatenate(y_val_true), label="True")
            plt.plot(sigmoid(np.concatenate(y_val_pred)), label="Predicted")
        else:
            plt.plot(np.concatenate(y_val_true), label="True")
            plt.plot(np.concatenate(y_val_pred), label="Predicted")
        plt.legend()
        plt.title(args.downstream_label)
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
    plt.imshow(pred_[batch_idx, :, :-1, 0].T, aspect="auto", interpolation="nearest")
    plt.title("Prediction")
    plt.gca().invert_yaxis()
    plt.clim(pred_[batch_idx, :, :-1, 0].min(), pred_[batch_idx, :, :, 0].max())
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(true_[batch_idx, :, :-1, 0].T, aspect="auto", interpolation="nearest")
    plt.title("True")
    plt.colorbar()
    plt.gca().invert_yaxis()
    a_ = true_[batch_idx, :, :, 0]
    a_max = np.max(a_[np.nonzero(a_)])
    plt.clim(true_[batch_idx, :, :-1, 0].min(), a_max
             )
    
    writer.add_figure('Pretrain Prediction', plt.gcf(), epoch)
    plt.close()

def report_tb_res(args, y_pred, y_true, epoch, writer, classification=False, train_=True):
    
    if train_:
        str_prefix = "train"
    else:
        str_prefix = "val"
    
    pred_ = np.concatenate(y_pred)
    true_ = np.concatenate(y_true)

    dict_res = {}

    if classification:
        metric_use = metrics.balanced_accuracy_score
        if args.downstream_label == "pkg_bk" or args.downstream_label == "all":
            if args.downstream_label == "pkg_bk":
                ba_bk = metric_use(true_, sigmoid(pred_)>0.5)
            else:
                ba_bk = metric_use(true_[:, 0], sigmoid(pred_[:, 0])>0.5)
            dict_res["ba_bk"] = ba_bk
            writer.add_scalar(f"{str_prefix}_ba_bk", ba_bk, epoch)
        if args.downstream_label == "pkg_dk" or args.downstream_label == "all":
            if args.downstream_label == "pkg_dk":
                ba_dk = metric_use(true_, sigmoid(pred_)>0.5)
            else:
                ba_dk = metric_use(true_[:, 1], sigmoid(pred_[:, 1])>0.5)
            dict_res["ba_dk"] = ba_dk
            writer.add_scalar(f"{str_prefix}_ba_dk", ba_dk, epoch)
        if args.downstream_label == "pkg_tremor":  #  or args.downstream_label == "all"
            if args.downstream_label == "pkg_tremor":
                ba_tremor = metric_use(true_, sigmoid(pred_)>0.5)
            else:
                ba_tremor = metric_use(true_[:, 2], sigmoid(pred_[:, 2])>0.5)
            dict_res["ba_tremor"] = ba_tremor
            writer.add_scalar(f"{str_prefix}_ba_tremor", ba_tremor, epoch)
    else:
        metric_use = np.corrcoef
        if args.downstream_label == "pkg_bk" or args.downstream_label == "all":
            if args.downstream_label == "pkg_bk":
                corr_bk = metric_use(true_, pred_)[0, 1]
            else:
                corr_bk = metric_use(true_[:, 0], pred_[:, 0])[0, 1]
            dict_res["corr_bk"] = corr_bk
            writer.add_scalar(f"{str_prefix}_corr_bk", corr_bk, epoch)
        if args.downstream_label == "pkg_dk" or args.downstream_label == "all":
            if args.downstream_label == "pkg_dk":
                corr_dk = metric_use(true_, pred_)[0, 1]
            else:
                corr_dk = metric_use(true_[:, 1], pred_[:, 1])[0, 1]
            dict_res["corr_dk"] = corr_dk
            writer.add_scalar(f"{str_prefix}_corr_dk", corr_dk, epoch)
        if args.downstream_label == "pkg_tremor" : # or args.downstream_label == "all"
            if args.downstream_label == "pkg_tremor":
                corr_tremor = metric_use(true_, pred_)[0, 1]
            else:
                corr_tremor = metric_use(true_[:, 2], pred_[:, 2])[0, 1]
            dict_res["corr_tremor"] = corr_tremor
            writer.add_scalar(f"{str_prefix}_corr_tremor", corr_tremor, epoch)

        metric_use = metrics.mean_absolute_error
        if args.downstream_label == "pkg_bk" or args.downstream_label == "all":
            if args.downstream_label == "pkg_bk":
                mae_bk = metric_use(true_, pred_)
            else:
                mae_bk = metric_use(true_[:, 0], pred_[:, 0])
            dict_res["mae_bk"] = mae_bk
            writer.add_scalar(f"{str_prefix}_mae_bk", mae_bk, epoch)
        if args.downstream_label == "pkg_dk" or args.downstream_label == "all":
            if args.downstream_label == "pkg_dk":
                mae_dk = metric_use(true_, pred_)
            else:
                mae_dk = metric_use(true_[:, 1], pred_[:, 1])
            dict_res["mae_dk"] = mae_dk
            writer.add_scalar(f"{str_prefix}_mae_dk", mae_dk, epoch)
        if args.downstream_label == "pkg_tremor":   #  or args.downstream_label == "all"
            if args.downstream_label == "pkg_tremor":
                mae_tremor = metric_use(true_, pred_)
            else:
                mae_tremor = metric_use(true_[:, 2], pred_[:, 2])
            dict_res["mae_tremor"] = mae_tremor
            writer.add_scalar(f"{str_prefix}_mae_tremor", mae_tremor, epoch)
    
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