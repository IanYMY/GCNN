import os
import yaml
import sys

# read out config(.yml)
configfile = sys.argv[1]
with open(configfile, 'r', encoding='utf-8') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)

os.environ['CUDA_VISIBLE_DEVICES'] = config['GNN_int_output']['cuda']
import torch
from torch_geometric.loader import DataLoader
import numpy as np
from utils import GNN_Dataset
from GNN_model import GNN

import h5py

# test set files
dataset_csv = config['GNN_int_output']['dataset_csv']
node_hdf = config['GNN_int_output']['node_hdf']
edge_attr_hdf = config['GNN_int_output']['edge_attr_hdf']
edge_ind_hdf = config['GNN_int_output']['edge_ind_hdf']
output_node_hdf = config['GNN_int_output']['output_node_hdf']


out_channels = config['GNN_int_output']['out_channels']
heads = config['GNN_int_output']['heads']
ratio = config['GNN_int_output']['ratio']
best_checkpoint_path = config['GNN_int_output']['best_checkpoint_path']
log_path = config['GNN_int_output']['log_path']
batch_size = 1


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

dataset = GNN_Dataset(dataset_csv, node_hdf, edge_attr_hdf, edge_ind_hdf)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True,
                        num_workers=4)

with torch.no_grad():
    features = []
    indices = []
    for batch_ind, batch_data in enumerate(dataloader):
        batch_data = batch_data.to(device)
        model(batch_data)

        # get intermediate outputs as new features
        feature = model.feature
        index = model.index
        features.append(feature.cpu())
        indices.append(index.cpu())

pdbids = list(np.loadtxt(dataset_csv, dtype=str, delimiter=",", skiprows=1, usecols=0))
badcomplexes = ['2r1w', '3sl0', '3sl1', '2y4a', '3skk', '5fom', '6hwz', '4ob0', '5agr', '4lv3',
                '3bho', '3zjt', '3mke', '6hx5', '3zju', '4wku', '3sjt', '4u0x', '6rtn', '5ee8', '5eec',
                '5agt', '3mnu', '4lv1', '5ags', '3fkv', '3ixg', '4lv2', '5wad', '6c8x']

# delete badcomplexes from pdbids
for badpdbid in badcomplexes:
    if badpdbid in pdbids:
        pdbids.remove(badpdbid)

f = h5py.File(node_hdf)

for i in range(len(pdbids)):
    pdbid = pdbids[i]
    index = indices[i]
    coordinate = f[pdbid][:, 0:4]
    coordinate = np.take(coordinate, index, axis=0)
    coordinate = torch.from_numpy(coordinate)
    feature = torch.tensor(features[i])
    new_feature = torch.concat((coordinate, feature), dim=1)
    new_feature = new_feature.numpy()
    affinity = f[pdbid].attrs['affinity']

    with h5py.File(output_node_hdf, 'a') as f1:
        keys = f1.keys()
        if pdbid not in keys:
            dset_node = f1.create_dataset(pdbid, data=new_feature, shape=new_feature.shape, dtype='float32',
                                          compression='lzf')
            dset_node.attrs['affinity'] = affinity

f.close()
