import os
import json
import argparse

import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from analyze.metrics import compute_cka, compute_weight_norm
from models import build_resnet, build_vgg, build_mlp

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mlp12", help="Model name")
parser.add_argument("--optimizer", type=str, default="sgd_momentum", help="Optimizer name")
parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name")
args = parser.parse_args()

MODEL_NAME = args.model
OPTIMIZER_NAME = args.optimizer
DATASET_NAME = args.dataset
result_path = f"results/{MODEL_NAME}_{DATASET_NAME}_{OPTIMIZER_NAME}/"
fig_path = os.path.join("figures", f"{MODEL_NAME}_{DATASET_NAME}_{OPTIMIZER_NAME}")
os.makedirs("figures", exist_ok=True)
os.makedirs(fig_path, exist_ok=True)

checkpoints_path = os.path.join(result_path, "checkpoints")
checkpoints = os.listdir(checkpoints_path)
checkpoints.sort(key=lambda x: int(x.split(".")[0][5:]))

num_epochs = int(checkpoints[-1].split(".")[0][5:])
print(f"Found {num_epochs} checkpoints")

if MODEL_NAME.startswith("resnet"):
    model = build_resnet(MODEL_NAME, 10)
elif MODEL_NAME.startswith("vgg"):
    model = build_vgg(MODEL_NAME, 10)
elif MODEL_NAME.startswith("mlp"):
    model = build_mlp(MODEL_NAME, 10)
else:
    raise ValueError(f"Unknown model name: {MODEL_NAME}")
layer_names = model.get_layer_names(); print(len(layer_names), layer_names)

label_names = []
if MODEL_NAME.startswith("resnet"):
    for i, name in enumerate(layer_names):
        label_names.append(f"conv_{i+1}")
elif MODEL_NAME.startswith("vgg"):
    for name in layer_names:
        if "conv_blocks" in name:
            label_names.append(f"conv_{int(name.split('.')[1]) + 1}")
        if "classifier" in name:
            label_names.append(f"linear_{int(name.split('.')[1]) + 1}")
elif MODEL_NAME.startswith("mlp"):
    for i, name in enumerate(layer_names):
        label_names.append(f"linear_{i+1}")

# ------ Loss and Accuracy Curves -----

checkpoint = torch.load(os.path.join(checkpoints_path, f"epoch{num_epochs}.pt"), map_location="cpu")

plt.figure(figsize=(8, 5))
plt.plot(checkpoint["train_history"]['loss'], label="train loss")
plt.plot(checkpoint["test_history"]['loss'], label="test loss")
plt.legend(title="")
plt.tight_layout()
plt.savefig(f"{fig_path}/loss.png")

plt.figure(figsize=(8, 5))
plt.plot(checkpoint["train_history"]['accuracy'], label="train acc")
plt.plot(checkpoint["test_history"]['accuracy'], label="test acc")
plt.legend(title="")
plt.tight_layout()
plt.savefig(f"{fig_path}/accuracy.png")

# ----- Tunnel Analysis -----
analysis = json.load(open(os.path.join(result_path, f"epoch{num_epochs}", "analysis", "analysis.json"), "r"))

final_acc = checkpoint["test_history"]['accuracy'][-1]
threshold_95 = 0.95 * final_acc
tunnel_95 = next((i for i, a in enumerate(analysis["probe_test_acc"]) if a >= threshold_95), len(analysis["layers"]) - 1)
threshold_98 = 0.98 * final_acc
tunnel_98 = next((i for i, a in enumerate(analysis["probe_test_acc"]) if a >= threshold_98), len(analysis["layers"]) - 1)

print("95% Tunnel Starts at", label_names[tunnel_95], f"layer {tunnel_95 + 1} ({tunnel_95/len(label_names)*100:.1f}%)")
print("98% Tunnel Starts at", label_names[tunnel_98], f"layer {tunnel_98 + 1} ({tunnel_98/len(label_names)*100:.1f}%)")

fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx() 

ax1.axvspan(tunnel_95 - 0.5, len(label_names),
            alpha=0.25, color="tab:gray")

sns.lineplot(analysis["test_numerical_rank"], linestyle='dashdot', color="tab:orange", ax=ax1, label="Test Numerical Rank", legend=False)
ax1.set_xlabel('Layers')
ax1.set_xticks(range(len(label_names)), labels=label_names, rotation=45, ha='right')
ax1.tick_params(axis='x', rotation=45)
ax1.set_ylabel('Numerical Rank')

sns.lineplot(analysis["probe_test_acc"], marker='o', color="tab:blue", ax=ax2, label="Probe Test Accuracy", legend=False)
ax2.set_ylabel('Probe Test Accuracy')

fig.tight_layout()
ax1.grid(True)

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, title="", loc="lower center")
plt.tight_layout()
plt.savefig(f"{fig_path}/tunnel_analysis.png")

# ----- Inter / Intra Variance -----

inters, intras = [], []
for inter, intra in analysis["test_variance"]:
    inters.append(inter)
    intras.append(intra)

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)

ax.axvspan(tunnel_95 - 0.5, len(label_names),
            alpha=0.25, color="tab:gray")

plt.plot(inters, label="Inter-class Variance", marker='o')
plt.plot(intras, label="Intra-class Variance", marker='o')

ax.set_xlabel('Layers')
ax.set_xticks(range(len(label_names)), labels=label_names, rotation=45, ha='right')
ax.set_ylabel('Variance')
plt.legend(title="")
plt.grid()
plt.tight_layout()
plt.savefig(f"{fig_path}/variance.png")

# # ----- CKA -----

# rows = []
# layer_names = analysis["layers"]
# for i, layer1 in enumerate(layer_names):
#     cols = []
#     for j, layer2 in enumerate(layer_names):
#         feat1 = torch.load(os.path.join(result_path, f"epoch{num_epochs}", "analysis", "features", f"test_feats_{layer1}.pt"), map_location="cpu")
#         feat2 = torch.load(os.path.join(result_path, f"epoch{num_epochs}", "analysis", "features", f"test_feats_{layer2}.pt"), map_location="cpu")
#         cka_value = compute_cka(feat1, feat2)
#         cols.append(cka_value)
#         del feat1, feat2
#     rows.append(cols)
# rows = np.array(rows)

# fig, ax = plt.subplots(figsize=(7, 7))

# sns.heatmap(
#     rows, ax=ax, cmap="magma", vmin=0, vmax=1,
#     square=True, cbar=True, cbar_kws={'shrink': 0.7},
#     xticklabels=layer_names, yticklabels=layer_names
# )
# ax.tick_params(axis="x", rotation=45)
# ax.tick_params(axis="y", rotation=0)
# plt.tight_layout()
# plt.savefig(f"{fig_path}/cka.png")

# # ----- Ranks per Epoch -----

# ranks = []

# for epoch in range(0, num_epochs + 1, 10):
#     ep = 1 if epoch == 0 else epoch
#     with open(os.path.join(result_path, f"epoch{ep}/analysis/analysis.json"), "r") as f:
#         analysis = json.load(f)
#     rank = analysis["test_numerical_rank"]
#     ranks.append(rank)

# from matplotlib.colors import Normalize
# fig = plt.figure(figsize=(8, 5))
# ax = fig.add_subplot(111)

# norm = Normalize(vmin=1, vmax=num_epochs)
# cmap = plt.get_cmap("YlGn")

# ax.axvspan(tunnel_95 - 0.5, len(label_names),
#             alpha=0.25, color="tab:gray")
# for i, rank in enumerate(ranks):
#     ep = 1 if i == 0 else (i) * 10
#     color = cmap(norm(ep))
#     plt.plot(rank, color=color, marker='o', label=f"Epoch {ep}")

# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# cbar = plt.colorbar(sm, ax=ax)
# cbar.set_label('Epoch', rotation=270, labelpad=15)
# plt.xticks(range(len(label_names)), labels=label_names, rotation=45, ha='right')
# plt.xlabel("Layers")
# plt.ylabel("Numerical Rank")
# plt.title("Numerical Rank Across Layers During Training")
# plt.grid()
# plt.tight_layout()
# plt.savefig(f"{fig_path}/ranks_per_epoch.png")

# # ----- Weight Differences per Epoch -----

# weight_diffs = []
# for epoch in tqdm(range(1, num_epochs + 1)):
#     if epoch % 10 != 0 and epoch != 1:
#         continue
    
#     checkpoint = torch.load(os.path.join(result_path, "checkpoints", f"epoch{epoch}.pt"), map_location="cpu")
#     curr_weights = checkpoint["model_state_dict"]

#     if epoch == 1:
#         prev_weights = curr_weights
#         continue

#     weight_diffs_epoch = []
#     for name in layer_names:
#         layer = name + ".weight"
#         w_prev = prev_weights[layer].float()
#         w_next = curr_weights[layer].float()
#         weight_diff = compute_weight_norm(w_prev, w_next)
#         weight_diffs_epoch.append(weight_diff)
#     weight_diffs.append(weight_diffs_epoch)
# data = np.array(weight_diffs)

# def plot_epoch_layer_heatmap(
#     data,
#     ax,
#     epochs=None,
#     layers=None,
#     cmap="viridis",
#     vmin=None,
#     vmax=None,
#     cbar=True,
#     linewidths=1,
#     cbar_ax=None,
#     linecolor="white",
#     x_ticks=None
# ):
#     data = np.asarray(data)
#     n_epochs, n_layers = data.shape

#     if epochs is None:
#         epochs = np.arange(n_epochs)
#     if layers is None:
#         layers = np.arange(n_layers)

#     sns.heatmap(
#         data,
#         ax=ax,
#         cmap=cmap,
#         vmin=vmin,
#         vmax=vmax,
#         yticklabels=epochs,
#         cbar=cbar,
#         cbar_ax=cbar_ax,
#         square=True,
#         linewidths=linewidths,
#         linecolor=linecolor,
#     )
#     ax.set_xlabel("")
#     ax.set_ylabel("")
#     if x_ticks is not None:
#         ax.set_xticks(range(len(x_ticks)), x_ticks)
#         ax.tick_params(axis="x", rotation=45)
#     else:
#         ax.set_xticks([])

#     return ax

# num_rows, num_cols = data.shape
# # Row labels
# row_labels = [
#     f"{i*10}-{(i+1)*10}" for i in range(num_rows)
# ]
# col_labels = label_names

# # Split data into groups

# groups = [
#     (data[:8], row_labels[:8], 0.2),
#     (data[8:12], row_labels[8:12], 0.2),
#     (data[12:], row_labels[12:], 0.2),
# ]

# fig, axes = plt.subplots(
#     len(groups), 1,
#     figsize=(10, 0.5*len(row_labels) + 0.5*len(groups) - 1),
#     gridspec_kw={
#         "height_ratios": [len(g[0]) for g in groups],
#         "hspace": 0.15
#     }
# )

# if len(groups) == 1:
#     axes = [axes]


# for i, (ax, (d, l, vmax)) in enumerate(zip(axes, groups)):
#     print(d.shape, l)

#     pos = ax.get_position()
#     cbar_ax = fig.add_axes([
#         pos.x1 - 0.06, pos.y0, 0.02, pos.height
#     ])
#     plot_epoch_layer_heatmap(
#         d,
#         ax=ax,
#         layers=label_names,
#         epochs=l,
#         cbar=True,
#         vmax=vmax,
#         cbar_ax=cbar_ax,
#         x_ticks=label_names if i == len(groups) - 1 else None,
#     )

# fig_w, fig_h = fig.get_size_inches()
# fig.supxlabel("Layers", fontsize=16, y=0.1 / fig_h)
# fig.supylabel("Epochs", fontsize=16, x=1.2 / fig_w)
# plt.savefig(f"{fig_path}/weight_diffs_per_epoch.png")