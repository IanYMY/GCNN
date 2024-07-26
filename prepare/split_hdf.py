import numpy as np
import h5py
import os
import sys
import yaml

# read out config(.yml)
configfile = sys.argv[1]
with open(configfile, 'r', encoding='utf-8') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)

csv_file = config['split_hdf']['csv_file']
node_file = config['split_hdf']['node_file']
path = config['split_hdf']['path']
dataset_name = config['split_hdf']['dataset_name']

file_path = os.path.join(path, dataset_name + '_set.hdf')

pdbids = list(np.loadtxt(csv_file, dtype=str, delimiter=",", skiprows=1, usecols=0))
badcomplexes = ['2r1w', '3sl0', '3sl1', '2y4a', '3skk', '5fom', '6hwz', '4ob0', '5agr', '4lv3',
                         '3bho', '3zjt', '3mke', '6hx5', '3zju', '4wku', '3sjt', '4u0x', '6rtn', '5ee8', '5eec',
                         '5agt', '3mnu', '4lv1', '5ags', '3fkv', '3ixg', '4lv2', '5wad', '6c8x']

# delete badcomplexes from pdbids
for badpdbid in badcomplexes:
    if badpdbid in pdbids:
        pdbids.remove(badpdbid)

with h5py.File(node_file, 'r') as f1:
    with h5py.File(file_path, 'a') as f2:
        for pdbid in pdbids:
            dset = f2.create_dataset(pdbid, data=f1[pdbid])
            dset.attrs['affinity'] = f1[pdbid].attrs['affinity']
