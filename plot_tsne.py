from matplotlib import pyplot as plt
import os
import pickle
import numpy as np
import sklearn

PATH_ = "out_save_debug/models_save_downstream/dict_res_emb.pkl"
PATH_FIGURES = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Dokumente/Decoding toolbox/Paper IEEE NER/figures'

with open(PATH_, 'rb') as f:
    model_trace = pickle.load(f)

embs = np.concatenate(model_trace["embs"])
embs = embs.reshape(embs.shape[0], -1)

labels = np.concatenate(model_trace["labels"])
hours = np.concatenate(model_trace["hours"])

# tsne
tsne = sklearn.manifold.TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)
tsne_results = tsne.fit_transform(embs)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels[:, 0], s=1.5)
cbar = plt.colorbar()
cbar.set_label("Wearable Bradykinesia Score")
plt.title("Bradykinesia")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.clim(20, 80)

plt.subplot(1, 2, 2)
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels[:, 1], s=1.5)
cbar = plt.colorbar()
cbar.set_label("Wearable Dyskinesia Score")
plt.title("Dyskinesia")
plt.clim(5, 20)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.savefig(os.path.join(PATH_FIGURES, "tsne_embs.pdf"))
plt.show()