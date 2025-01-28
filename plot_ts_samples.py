from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os

if __name__ == "__main__":

    PATH_DATA = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/ts_transformer"

    data = np.load(os.path.join(PATH_DATA, "all_arr_series.npy"))
    labels = pd.read_csv(os.path.join(PATH_DATA, "all_label_series.csv"))
    labels["pkg_dt"] = pd.to_datetime(labels["pkg_dt"])
    labels["pkg_h"] = labels["pkg_dt"].dt.hour
    labels["sub_id"] = pd.Categorical(labels["sub"]).codes
    labels["pkg_bk_class"] = labels["pkg_bk"] > 50
    labels["pkg_tremor_class"] = labels["pkg_tremor"] > 0
    labels["pkg_dk_class"] = labels.groupby("sub")["pkg_dk"].transform(lambda x: (x - x.min()) / (x.max() - x.min())) > 0.02

    # plot images in pdf
    from matplotlib.backends.backend_pdf import PdfPages

    pdf = PdfPages('ts_samples.pdf')
    for idx_outer in np.arange(0, data.shape[0]-10, 10):
        plt.figure(figsize=(10, 3))
        print(f"Plotting {idx_outer}")
        for idx_ts_range in range(10):
            plt.subplot(2, 5, idx_ts_range + 1)
            plt.imshow(data[idx_outer+idx_ts_range, :, :, 0].T, cmap="viridis", aspect="auto", interpolation="nearest")
            #plt.colorbar()
            plt.clim(-35, -25)
            plt.title(f"{idx_outer+idx_ts_range}")
            #plt.xlabel("time")
            #plt.ylabel("frequency")
            # flip y axis
            plt.gca().invert_yaxis()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    pdf.close()

