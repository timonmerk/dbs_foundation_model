from matplotlib import pyplot as plt
import numpy as np

PATH_DATA = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/ts_transformer"

all_arr_series = np.load(f"{PATH_DATA}/all_arr_series.npy")

print(all_arr_series.shape)
sp_mean = all_arr_series.mean(axis=(0, 1, 3))

plt.plot(sp_mean, label="mean")
log_f = np.log(np.arange(1, 126 + 1)) + sp_mean.mean()
plt.plot(log_f, label="log")
plt.plot((sp_mean + log_f)/2, label="mean between log and sp. mean")
plt.legend()
plt.show()