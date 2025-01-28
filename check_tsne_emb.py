import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

emb = np.load("embs/embs.npy")
# reshape and keep first dim
emb = emb.reshape(emb.shape[0], -1)
y = np.load("embs/labels.npy")

# compute tsne
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=100, n_iter=300)
emb_tsne = tsne.fit_transform(emb)
label_labels = ["bk", "dk", "tremor", "h", "sub"]

plt.figure(figsize=(7, 4))
for idx_plt in range(5):

    plt.subplot(2, 3, idx_plt + 1)
    if label_labels[idx_plt] == "sub":
        cmap_ = "jet"
    else:
        cmap_ = "viridis"
    plt.scatter(emb_tsne[:, 0], emb_tsne[:, 1], c=y[:, idx_plt], s=2, alpha=0.8, cmap=cmap_)
    plt.title(label_labels[idx_plt])
    
    if label_labels[idx_plt] == "bk":
        plt.clim(30, 80)
    elif label_labels[idx_plt] == "dk":
        plt.clim(0, 20)
    elif label_labels[idx_plt] == "tremor":
        plt.clim(0, 20)
    plt.colorbar()
    
plt.tight_layout()
plt.show(block=True)
