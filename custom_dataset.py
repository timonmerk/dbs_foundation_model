import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

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