import os
import yaml
import sys

# read out config(.yml)
configfile = sys.argv[1]
with open(configfile, 'r', encoding='utf-8') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)

os.environ['CUDA_VISIBLE_DEVICES'] = config['test_GNN']['cuda']
import torch
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
from GNN_model import GNN
from utils import GNN_Dataset


# test set files
test_set = config['test_GNN']['test_set']
test_csv = config['test_GNN']['test_csv']
node_hdf = config['test_GNN']['node_hdf']
edge_attr_hdf = config['test_GNN']['edge_attr_hdf']
edge_ind_hdf = config['test_GNN']['edge_ind_hdf']


out_channels = config['test_GNN']['out_channels']
heads = config['test_GNN']['heads']
ratio = config['test_GNN']['ratio']
best_checkpoint_path = config['test_GNN']['best_checkpoint_path']
log_path = config['test_GNN']['log_path']
batch_size = config['test_GNN']['batch_size']


device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = GNN(out_channels, heads, ratio)
model.to(device)

best_chehckpoint_dict = torch.load(best_checkpoint_path)
model_state_raw = best_chehckpoint_dict['model_state_dict']

# cut 'module.' out of the key
model_state = {}
for k, v in model_state_raw.items():
    new_k = k.replace('module.', '') if 'module' in k else k
    model_state[new_k] = v

model.load_state_dict(model_state)

test_dataset = GNN_Dataset(test_csv, node_hdf, edge_attr_hdf, edge_ind_hdf)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True,
                             num_workers=4)

with torch.no_grad():
    y_true = np.zeros((len(test_dataset),), dtype=np.float32)
    y_pred = np.zeros((len(test_dataset),), dtype=np.float32)
    for batch_ind, batch_data in enumerate(test_dataloader):
        batch_data = batch_data.to(device)
        y_ = model(batch_data)
        y = batch_data.y
        y_true[batch_ind * batch_size:batch_ind * batch_size + batch_size] = y.cpu().float().data.numpy()[:, 0]
        y_pred[batch_ind * batch_size:batch_ind * batch_size + batch_size] = y_.cpu().float().data.numpy()[:, 0]

r2 = r2_score(y_true=y_true, y_pred=y_pred)
mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
mse = mean_squared_error(y_true, y_pred)
pearsonr = stats.pearsonr(y_true.reshape(-1), y_pred.reshape(-1))
spearmanr = stats.spearmanr(y_true.reshape(-1), y_pred.reshape(-1))

print('rmse:{:0.4f}\t r2: {:0.4f}\t pearsonr: {:0.4f}\t spearmanr: {:0.4f}\t mae: {:0.4f}'
      .format(mse ** (1 / 2), r2, float(pearsonr[0]), float(spearmanr[0]), float(mae)))

# output csv file
test_dict = {'y_true': y_true, 'y_pred': y_pred}
train_df = pd.DataFrame(test_dict)
csv_path = os.path.join(log_path, test_set + '_value.csv')
train_df.to_csv(csv_path, index=False)

# output png file
fig, ax = plt.subplots()
ax.scatter(y_true, y_pred, s=3, c=['#1f77b4'])

a, b = np.polyfit(y_true, y_pred, deg=1)
y_est = a * y_true + b
ax.plot(y_true, y_est, '-', c='cornflowerblue')

ax.set_aspect('equal', 'box')
ax.set(xlim=(0, 13), ylim=(0, 13))

plt.title("CASF-2016", size=16)
plt.xlabel("Experimental pK", size=12)
plt.ylabel("Predicted pK", size=12)

ax.annotate(f"Rp={float(pearsonr[0]) : .3f}", (1.5, 12))
ax.annotate(f"RMSE={mse ** (1 / 2) : .3f}", (1.5, 11.5))

png_path = os.path.join(log_path, test_set + '_value.png')
plt.savefig(png_path, dpi=300)
