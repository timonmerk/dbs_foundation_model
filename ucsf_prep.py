import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
import tqdm
from scipy import signal
from joblib import Parallel, delayed

def check_missing_data(df):
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = None
    return df

PATH_PARQUET = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/parquet'
PATH_OUT = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/ts_transformer'
subs = np.unique([f[:6] for f in os.listdir(PATH_PARQUET)])

def run_sub(sub):
    sub_files = np.array(
        [
            (f, int(f[f.find(sub) + len(sub) + 1 : f.find(".parq")]))
            for f in os.listdir(PATH_PARQUET)
            if sub in f
        ]
    )
    sort_idx = np.argsort(sub_files[:, 1].astype(int))
    files_sorted = sub_files[sort_idx, 0]
    cnt_ID = 0
    arr_concat = []
    times_ = []
    for f in tqdm.tqdm(files_sorted):#[-20:]:
        df = pd.read_parquet(os.path.join(PATH_PARQUET, f))
        # df = df.astype(object)
        # df.set_index("timestamp", inplace=True)
        df.index = pd.to_datetime(df.index)
        df_r = df.resample("4ms").ffill(limit=1)
        # find indices of 10 second intervals
        # set the start value to rounded full 10 s
        start_ = df_r.index[0].ceil("10s")
        idx_10s = pd.date_range(start=start_, freq="10s", end=df_r.index[-1])

        for idx, idx_time in enumerate(idx_10s, start=1):
            if idx == idx_10s.shape[0]:
                break
            t_low = idx_10s[idx - 1]
            t_high = idx_10s[idx]
            df_r_ = df_r.loc[t_low:t_high]

            df_r_f = check_missing_data(df_r_)
            if df_r_f.sum().sum() == 0:
                continue

            # sum elements in each column that is not None or NaN
            column_sums = df_r_f.apply(lambda x: x.dropna().abs().sum())

            # select the four columns with the highest sums
            top_columns = column_sums.nlargest(4).index.sort_values()[:2]  # select only dbs
            if (column_sums[top_columns] == 0).any():
                continue
            df_ = df_r_f[top_columns]

            # check if there is a single columns that is not NaN
            # indexes of NaN values

            if df_.isnull().values.sum() == 0:
                # print(t_high)
                arr_ = np.array(df_.values)

                f, Pxx = signal.welch(arr_, fs=250, nperseg=250, axis=0)
                t_ = t_low + pd.Timedelta(5, "s")
                arr_concat.append(np.log(Pxx))
                times_.append(t_)
    np.save(os.path.join(PATH_OUT, sub + "_welch.npy"), np.array(arr_concat))
    np.save(os.path.join(PATH_OUT, sub + "_times.npy"), np.array(times_))

                
    

if __name__ == "__main__":

    # parallelize the function
    Parallel(n_jobs=len(subs))(delayed(run_sub)(sub) for sub in subs)

    #for sub in subs:
    #    run_sub(sub)