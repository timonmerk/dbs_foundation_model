import os
import pickle
from matplotlib import pyplot as plt
import pandas as pd

PATH_traces = "out_save_hour/models_save_downstream/regression"
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
    epoch = list(model_trace.keys())[-1]
    per_bk = model_trace[epoch]["corr_bk"]
    per_.append({
        "sub": sub,
        "per_bk": model_trace[epoch]["corr_bk"],
        "per_dk": model_trace[epoch]["corr_dk"],
        "per_tremor": model_trace[epoch]["corr_tremor"],
    })
df = pd.DataFrame(per_)

df.per_bk.plot.bar()
df.per_dk.plot.bar()
df.per_tremor.plot.bar()

PATH_EVENTS = "out_save_hour/runs"
subs = [f for f in os.listdir(PATH_EVENTS) if "rcs" in f]
sub = subs[0]
f_events = f_path = os.path.join(PATH_EVENTS, sub, os.listdir(os.path.join(PATH_EVENTS, sub))[2])

sub = "rcs02"
f_path = os.path.join(PATH_traces, sub, "dict_res.pkl")
with open(f_path, 'rb') as f:
    model_trace = pickle.load(f)



epoch = list(model_trace.keys())[-1]
plt.plot(model_trace[epoch]["y_pred"][:, 0], label="y_pred")
plt.plot(model_trace[epoch]["y_true"][:, 0], label="y_true")
plt.legend()
plt.show()



from tensorboard.backend.event_processing import event_accumulator

ea = event_accumulator.EventAccumulator(f_events)
ea.Reload()

ea.scalars.Keys()

ea.scalars.Items('train_pretrain_loss')

scalar_data = {}
for tag in ea.scalars.Keys():
    scalar_events = ea.scalars.Items(tag)  # Get all events for this scalar
    scalar_data[tag] = [(event.step, event.value) for event in scalar_events]

import matplotlib.pyplot as plt
import numpy as np

scalar_data["train_pretrain_loss"]
plt.figure()
plt.plot(*zip(*scalar_data["train_pretrain_loss"]), label="train_pretrain_loss")
plt.plot(*zip(*scalar_data["val_pretrain_loss"]), label="val_pretrain_loss")
# make log scale
plt.yscale('log')
plt.show()
