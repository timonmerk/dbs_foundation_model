import pandas as pd
import itertools
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


df = pd.read_csv("joint_df.csv")

lr_list = ["0.1", "0.01", "0.001", "0.0001"]
d_model_list = [8, 16, 32, 64, 128, 256]
apply_log_scaling_list = [True, False]
dim_feedforward_list = [8, 16, 32, 64, 128, 256]
combinations = list(itertools.product(lr_list, d_model_list, apply_log_scaling_list, dim_feedforward_list))

df_res = []
for cond in tqdm(df["cond"].unique()):
    df_cond = df[df["cond"] == cond]
    for sub in df_cond["sub"].unique():
        df_sub = df_cond[df_cond["sub"] == sub]
        epoch_last = np.sort(df_sub["epoch"].unique())[-1]
        df_sub = df_sub[df_sub["epoch"] == epoch_last][["corr_bk", "corr_dk"]]
        lr = combinations[cond][0]
        d_model = combinations[cond][1]
        apply_log_scaling = combinations[cond][2]
        dim_feedforward = combinations[cond][3]
        df_res.append({
            "lr": lr,
            "d_model": d_model,
            "apply_log_scaling": apply_log_scaling,
            "dim_feedforward": dim_feedforward,
            "corr_bk": df_sub["corr_bk"].values[0],
            "corr_dk": df_sub["corr_dk"].values[0],
            "sub": sub,
        })
df_res = pd.DataFrame(df_res)
# average over sub
df_res_ = df_res.groupby(["lr", "d_model", "apply_log_scaling", "dim_feedforward"])[["corr_bk", "corr_dk"]].mean().reset_index()

bk_per_sorted = df_res_.sort_values(by=["corr_bk"], ascending=True)["corr_bk"].values
dk_per_sorted = df_res_.sort_values(by=["corr_dk"], ascending=True)["corr_dk"].values
plt.plot(bk_per_sorted, label="bk")
plt.plot(dk_per_sorted, label="dk")
plt.xlabel("Parameter combination")
plt.ylabel("Pearson correlation coefficient")
plt.legend()
plt.show()


# Plot the names of the best 10 parameters
best_10 = df_res_.sort_values(by=["corr_bk"], ascending=False).head(10)
best_10 = best_10.sort_values(by=["corr_bk"], ascending=True)
corr_bk = best_10["corr_bk"].values
best_10 = best_10[["lr", "d_model", "apply_log_scaling", "dim_feedforward"]].values
best_10 = [f"lr: {x[0]}, d_model: {x[1]}, apply_log_scaling: {x[2]}, dim_feedforward: {x[3]}" for x in best_10]

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.barh(np.arange(len(best_10)), corr_bk, label="bk")
plt.yticks(np.arange(len(best_10)), best_10, rotation=0)
plt.ylabel("Parameter combination")
plt.title("Bradykinesia")

plt.subplot(2, 1, 2)
best_10 = df_res_.sort_values(by=["corr_dk"], ascending=False).head(10)
best_10 = best_10.sort_values(by=["corr_dk"], ascending=True)
corr_dk = best_10["corr_dk"].values
best_10 = best_10[["lr", "d_model", "apply_log_scaling", "dim_feedforward"]].values
best_10 = [f"lr: {x[0]}, d_model: {x[1]}, apply_log_scaling: {x[2]}, dim_feedforward: {x[3]}" for x in best_10]
plt.barh(np.arange(len(best_10)), corr_dk, label="dk")
plt.yticks(np.arange(len(best_10)), best_10, rotation=0)
plt.ylabel("Parameter combination")
plt.title("Dyskinesia")
plt.suptitle("Best 10 parameter combinations")
plt.tight_layout()
plt.savefig("best_10_params.pdf")
plt.show()


pairs = [("dim_feedforward", "lr"),
         ("dim_feedforward", "d_model"),
         ("apply_log_scaling", "lr"),
         ("apply_log_scaling", "d_model"),
         ("apply_log_scaling", "dim_feedforward"),
         ("d_model", "lr")]

pp = PdfPages('gs_analysis.pdf')
for pair in pairs:
    hp_1, hp_2 = pair
    df_lr_dmodel = df_res_.groupby([hp_1, hp_2])[["corr_bk", "corr_dk"]].mean().reset_index()
    fig = plt.figure(figsize=(8, 4))
    for idx_, label_ in enumerate(["corr_bk", "corr_dk"]):
        plt.subplot(1, 2, idx_ + 1)
        plt.imshow(df_lr_dmodel.pivot(index=hp_1, columns=hp_2, values=label_), cmap="viridis", interpolation="nearest")
        plt.colorbar()
        plt.ylabel(hp_1)
        plt.yticks(np.arange(len(df_lr_dmodel[hp_1].unique())), df_lr_dmodel[hp_1].unique())
        plt.xticks(np.arange(len(df_lr_dmodel[hp_2].unique())), df_lr_dmodel[hp_2].unique())
        plt.xlabel(hp_2)
    plt.suptitle(f"{hp_1} vs {hp_2}")
    plt.tight_layout()
    pp.savefig(fig)
    plt.close()
pp.close()
#plt.show()

