import pandas as pd
import os

PATH_DATA = "out_log"

d_all = []
for dir in os.listdir(PATH_DATA):
    if os.path.exists(os.path.join(PATH_DATA, dir, "results_regression.csv")):
        df_read = pd.read_csv(os.path.join(PATH_DATA, dir, "results_regression.csv"))
        df_read["cond"] = dir
        d_all.append(df_read)
all_ = pd.DataFrame(d_all)
all_.to_csv("joint_df.csv")