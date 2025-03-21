import os
import pickle
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from tensorboard.backend.event_processing import event_accumulator

PATH_FIGURES = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Dokumente/Decoding toolbox/Paper IEEE NER/figures'

PATH_traces = "out_cluster/regression"
subs = os.listdir(PATH_traces)

per_ = []
for sub in subs:
    f_path = os.path.join(PATH_traces, sub, "dict_res.pkl")
    try:
        with open(f_path, 'rb') as f:
            model_trace = pickle.load(f)
    except:
        print(f"Error with {sub}")
        continue

    for epoch in list(model_trace.keys()):
        per_bk = model_trace[epoch]["corr_bk"]
        last_epoch = True if epoch == list(model_trace.keys())[-1] else False
        per_.append({
            "sub": sub,
            "pkg_type" : "bk",
            "per" : model_trace[epoch]["corr_bk"],
            "epoch" : epoch,
            "mae" : model_trace[epoch]["mae_bk"],
            "last_epoch" : last_epoch
        })
        per_.append({
            "sub": sub,
            "pkg_type" : "dk",
            "per" : model_trace[epoch]["corr_dk"],
            "epoch" : epoch,
            "mae" : model_trace[epoch]["mae_dk"],
            "last_epoch" : last_epoch
        })
df = pd.DataFrame(per_)


# PLOT ind. prediction traces
# save each plot in a pdf
from matplotlib.backends.backend_pdf import PdfPages

pdf_path = os.path.join(PATH_FIGURES, "ind_traces.pdf")

with PdfPages(pdf_path) as pdf:
    for sub in np.sort(subs):
        f_path = os.path.join(PATH_traces, sub, "dict_res.pkl")
        try:
            with open(f_path, 'rb') as f:
                model_trace = pickle.load(f)
        except:
            print(f"Error with {sub}")
            continue

        epoch = list(model_trace.keys())[-1]
        per_bk = model_trace[epoch]["corr_bk"]
        plt.figure()
        plt.subplot(2, 1, 1)
        pr_ = model_trace[epoch]["y_pred"][:, 0]
        pr_ = stats.zscore(pr_)
        true_ = model_trace[epoch]["y_true"][:, 0]
        true_ = stats.zscore(true_)
        plt.plot(pr_, label="y_pred")
        plt.plot(true_, label="y_true")
        plt.legend()
        plt.title(f"{sub} - Bradykinesia corr = {per_bk:.2f}")
        plt.subplot(2, 1, 2)
        per_dk = model_trace[epoch]["corr_dk"]
        pr_ = model_trace[epoch]["y_pred"][:, 1]
        pr_ = stats.zscore(pr_)
        true_ = model_trace[epoch]["y_true"][:, 1]
        true_ = stats.zscore(true_)
        plt.plot(pr_, label="y_pred")
        plt.plot(true_, label="y_true")
        plt.legend()
        plt.title(f"{sub} - Dyskinesia corr = {per_dk:.2f}")
        pdf.savefig()
        plt.close()




PLT_ = True

if PLT_:
    plt.figure()
    sns.boxplot(data=df.query("last_epoch == True"), x="pkg_type", y="per", boxprops=dict(alpha=.3), showmeans=True)
    sns.swarmplot(data=df.query("last_epoch == True"), x="pkg_type", y="per", color=".25")
    plt.ylabel("Correlation coefficient")
    plt.savefig(os.path.join(PATH_FIGURES, "per_boxplot.pdf"))
    plt.show()

    plt.figure()
    plt.subplot(1, 2, 1)
    for sub in df["sub"].unique():
        df_sub = df.query(f"sub == '{sub}' and pkg_type == 'bk'")
        plt.plot(df_sub["epoch"], df_sub["per"], label=sub, color="gray", alpha=0.5)

    per__ = df.query("pkg_type == 'bk'").groupby("epoch")["per"].mean() # .rolling(window=2).mean()
    plt.plot(df.query("pkg_type == 'bk'")["epoch"].unique(), per__, label="mean", color="black", linewidth=2.5)
    plt.title("Bradykinesia")
    plt.xlabel("Epoch")
    plt.ylabel("Correlation coefficient")
    plt.subplot(1, 2, 2)
    for sub in df["sub"].unique():
        df_sub = df.query(f"sub == '{sub}' and pkg_type == 'dk'")
        plt.plot(df_sub["epoch"], df_sub["per"], label=sub, color="gray", alpha=0.5)
    per__ = df.query("pkg_type == 'dk'").groupby("epoch")["per"].mean()
    plt.plot(df.query("pkg_type == 'dk'")["epoch"].unique(), per__, label="mean", color="black", linewidth=2.5)
    plt.title("Dyskinesia")
    plt.xlabel("Epoch")
    plt.ylabel("Correlation coefficient")
    plt.savefig(os.path.join(PATH_FIGURES, "finetuning_per.pdf"))
    plt.show()

PATH_EVENTS = "out_cluster/runs"
subs = [f for f in os.listdir(PATH_EVENTS) if "rcs" in f]
l_ = []
for sub in subs:
    tf_event_files = os.listdir(os.path.join(PATH_EVENTS, sub))
    for f in tf_event_files:
        f_path = os.path.join(PATH_EVENTS, sub, f)
        ea = event_accumulator.EventAccumulator(f_path)
        ea.Reload()
        keys_ = ea.scalars.Keys()
        if "train_pretrain_loss" not in keys_:
            continue
        
        scalar_data = {}
        for tag in ea.scalars.Keys():
            scalar_events = ea.scalars.Items(tag)  # Get all events for this scalar
            scalar_data[tag] = [(event.step, event.value) for event in scalar_events]
            for step, value in scalar_data[tag]:
                l_.append({
                    "sub": sub,
                    "tag": "val" if "val" in tag else "train",
                    "epoch": step,
                    "mae": value
                })

df_all = pd.DataFrame(l_)

plt.figure()
plt.subplot(1, 2, 1)
for sub in df_all["sub"].unique():
    df_sub = df_all.query(f"sub == '{sub}' and tag == 'train'")
    plt.plot(df_sub["epoch"], df_sub["mae"], label=sub, color="gray", alpha=0.5)
per__ = df_all.query("tag == 'train'").groupby("epoch")["mae"].mean()
plt.plot(df_all.query("tag == 'train'")["epoch"].unique(), per__, label="mean", color="black", linewidth=2.5)
plt.yscale('log')
plt.ylim([120, 300])
plt.ylabel("MAE")
plt.xlabel("Epoch")
plt.title("Train set")

plt.subplot(1, 2, 2)
for sub in df_all["sub"].unique():
    df_sub = df_all.query(f"sub == '{sub}' and tag == 'val'")
    plt.plot(df_sub["epoch"], df_sub["mae"], label=sub, color="gray", alpha=0.5)
per__ = df_all.query("tag == 'val'").groupby("epoch")["mae"].mean()
plt.plot(df_all.query("tag == 'val'")["epoch"].unique(), per__, label="mean", color="black", linewidth=2.5)
# make log scale
plt.yscale('log')
plt.ylim([2, 200])
plt.ylabel("MAE")
plt.xlabel("Epoch")
plt.title("Validation set")
plt.savefig(os.path.join(PATH_FIGURES, "epochs_train.pdf"))
plt.show()




plt.figure()    
sub = "rcs02"
f_path = os.path.join(PATH_traces, sub, "dict_res.pkl")
with open(f_path, 'rb') as f:
    model_trace = pickle.load(f)



epoch = list(model_trace.keys())[-1]
plt.plot(model_trace[epoch]["y_pred"][:, 0], label="y_pred")
plt.plot(model_trace[epoch]["y_true"][:, 0], label="y_true")
plt.legend()
plt.show()






ea.scalars.Keys()

ea.scalars.Items('train_pretrain_loss')

import matplotlib.pyplot as plt
import numpy as np

scalar_data["train_pretrain_loss"]
plt.figure()
plt.plot(*zip(*scalar_data["train_pretrain_loss"]), label="train_pretrain_loss")
plt.plot(*zip(*scalar_data["val_pretrain_loss"]), label="val_pretrain_loss")
# make log scale
plt.yscale('log')
plt.show()
