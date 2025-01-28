import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from matplotlib import pyplot as plt

PATH_PKG = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/pkg_data'
PATH_IN = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/ts_transformer'
subs = np.unique([f[:6] for f in os.listdir(PATH_PKG) if "rcs" in f])

def map_sub(sub):

    ts_ = np.load(os.path.join(PATH_IN, f"{sub}_times.npy"), allow_pickle=True)
    ts_ = pd.Series(ts_, name="time_stamps")
    data = np.load(os.path.join(PATH_IN, f"{sub}_welch.npy"))

    df_pkg = pd.read_csv(os.path.join(PATH_PKG, f"{sub}_pkg.csv"))
    # map df_pkg pkg_dt to ts_ timestamps
    df_pkg["pkg_dt"] = pd.to_datetime(df_pkg["pkg_dt"])
    # smooth pkg data with plus minus 5 min
    df_pkg["pkg_bk"] = df_pkg["pkg_bk"].rolling(window=5).mean()
    df_pkg["pkg_dk"] = df_pkg["pkg_dk"].rolling(window=5).mean()
    df_pkg["pkg_tremor"] = df_pkg["pkg_tremor"].rolling(window=5).mean()

    # plt.figure()
    # plt.plot(df_pkg["pkg_bk"].values, label="bk")
    # plt.plot(df_pkg["pkg_bk_smooth"].values, label="bk smooth")
    # plt.legend()
    # plt.show(block=True)

    d_out_pkg = []
    d_out_ = []


    for idx, row in tqdm(df_pkg.iterrows()):
        # check that pkg_bk, pkg_dk and pkg_tremor are not NaN
        if np.isnan(row["pkg_bk"]) or np.isnan(row["pkg_dk"]) or np.isnan(row["pkg_tremor"]):
            continue
        ts = row["pkg_dt"]
        idx_ = np.where((ts - pd.Timedelta("60s") <= ts_) & (ts_ <= ts + pd.Timedelta("60s")))[0]
        if len(idx_) > 0:
            d_out_pkg.append(row)
            d_out_.append(data[idx_].mean(axis=0))

    df_out_pkg = pd.DataFrame(d_out_pkg).reset_index(drop=True)[["pkg_dt", "pkg_bk", "pkg_dk", "pkg_tremor"]]
    d_out_arr = np.array(d_out_)

    d_series_ = []
    d_series_time = []
    for idx, row in df_out_pkg.iterrows():
        # check how many values are available in the previous 30 min
        ts = row["pkg_dt"]
        idx_ = np.where((df_out_pkg["pkg_dt"] >= (ts - pd.Timedelta("30m"))) & (df_out_pkg["pkg_dt"] < ts))[0]
        if len(idx_) >= 15:
            d_series_.append(data[idx_])
            d_series_time.append(row)
    d_arr_series = np.array(d_series_)
    d_series_time_ = pd.DataFrame(d_series_time)
    return d_arr_series, d_series_time_

if __name__ == "__main__":
    
    out_all_arr = []
    out_all_label = []
    for sub in subs:
        d_arr_series, d_series_time_ = map_sub(sub)
        d_series_time_["sub"] = sub
        out_all_arr.append(d_arr_series)
        out_all_label.append(d_series_time_)
    out_all_arr = np.concatenate(out_all_arr)
    out_all_label = pd.concat(out_all_label).reset_index(drop=True)

    samples_omit = [
        283, 284, 285, 286,
        921, 922, 923, 924, 925, 926,
        945, 946, 947, 948, 949, 950, 951, 952,
        970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 
        990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 
        1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 
        1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033,
        1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 
        1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232, 1233, 1234, 
        1235, 1236, 1237, 1238, 1239, 1240, 1241, 1242, 1243, 1244,
        1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1259, 1260, 1261, 1262, 1263, 1264,
        1604, 1605, 1606, 1607, 1608, 1609,
        1624, 1625, 1626, 1627, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 1635,
        2363, 2364,
        2468, 2469, 2470, 2471, 2472, 2473, 2474, 2475, 2476, 2477, 2478, 2479, 2480,
        2491, 2492, 2493, 2494,
        3115,
        3208, 3209, 3210, 3211,
        4431, 4432, 4433, 4434, 4435, 4436,
        4549, 4550, 4551, 4552, 4553, 4554, 4555, 4556, 4557, 4558, 4559, 4560, 4561, 4562, 4563, 4564, 
        4565, 4566, 4567, 4568,
        9015, 9016, 9017, 9018, 9019, 9020, 9021, 9022, 9023, 9024, 9025, 9026, 9027, 9028, 9029, 9030, 9031,
        9078, 9079, 9080, 9081, 9082, 9083, 9084, 9085, 9086, 9087, 9088, 9089, 9090, 9091, 9092,
        9540, 9541, 9542, 9543, 9544, 9545, 9546, 9547, 9548,
        9706, 9707, 9708, 9709, 9710, 9711, 9712, 9713, 9714, 9715, 9716, 9717, 9718, 9719, 9720, 9721, 
        9722, 9723, 9724, 9725,
        9769, 9770, 9771, 9772, 9773, 9774, 9775, 9776, 9777, 9778, 9779, 9780, 9781, 9782, 9783,
        9786, 9787, 9788, 9789, 9790, 9791, 9792, 9793, 9794, 9795, 9796, 9797, 9798, 9799, 9800, 9801, 
        9802, 9803,
        9920, 9921
    ]

    # plt.imshow(out_all_arr[9027, :, :, 0].T, cmap="viridis", aspect="auto", interpolation="nearest")
    # plt.colorbar()
    # plt.clim(-35, -25)
    # plt.title("1237")
    # plt.gca().invert_yaxis()
    # plt.show(block=True)

    out_all_arr = np.delete(out_all_arr, samples_omit, axis=0)
    out_all_label = out_all_label.drop(samples_omit).reset_index(drop=True)
    
    np.save(os.path.join(PATH_IN, "all_arr_series.npy"), out_all_arr)
    out_all_label.to_csv(os.path.join(PATH_IN, "all_label_series.csv"), index=False)
        
    