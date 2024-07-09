import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
import numpy as np
import h5py



# normalize charges and vdW radius
def normalization():
    charges = []
    vdws = []
    f = h5py.File('/public/BioPhys/yym/yymbin/GCNN/dataset/cutoff8/node_data.h5', 'r')
    for key in f.keys():
        charge = np.array(f[key][:, 16]).flatten()
        vdw = np.array(f[key][:, 23]).flatten()
        charges = np.concatenate([charges, charge])
        vdws = np.concatenate([vdws, vdw])

    m_charge = charges.mean()
    std_charge = charges.std()
    m_vdw = vdws.mean()
    std_vdw = vdws.std()
    print('charges: mean=%s, sd=%s' % (m_charge, std_charge))
    print('use sd to normalize charges')
    print('vdws: mean=%s, sd=%s' % (m_vdw, std_vdw))
    print('use mean to normalize vdws')
    print('-------------------------------------------')
    return m_charge, std_charge, m_vdw, std_charge


class GNN_Dataset(Dataset):

    def __init__(self, csv_file, node_file, edge_attr_file, edge_ind_file):
        super(GNN_Dataset, self).__init__()
        self.m_charge, self.std_charge, self.m_vdw, self.std_vdw = normalization()
        self.csv_file = csv_file
        self.node_file = node_file
        self.edge_attr_file = edge_attr_file
        self.edge_ind_file = edge_ind_file
        self.pdbids = list(np.loadtxt(csv_file, dtype=str, delimiter=",", skiprows=1, usecols=0))
        self.badcomplexes = ['2r1w', '3sl0', '3sl1', '2y4a', '3skk', '5fom', '6hwz', '4ob0', '5agr', '4lv3',
                             '3bho', '3zjt', '3mke', '6hx5', '3zju', '4wku', '3sjt', '4u0x', '6rtn', '5ee8', '5eec',
                             '5agt', '3mnu', '4lv1', '5ags', '3fkv', '3ixg', '4lv2', '5wad', '6c8x']
        # delete badcomplexes from pdbids
        for self.badpdbid in self.badcomplexes:
            if self.badpdbid in self.pdbids:
                self.pdbids.remove(self.badpdbid)

    def __len__(self):
        return len(self.pdbids)

    def __getitem__(self, idx):
        pdbid = self.pdbids[idx]
        f1 = h5py.File(self.node_file)
        f2 = h5py.File(self.edge_attr_file)
        f3 = h5py.File(self.edge_ind_file)

        x = f1[pdbid][:, 4:]
        x[:, 14] = x[:, 14]/self.std_charge  # normalization of charges
        x[:, 19] = x[:, 19]/self.m_vdw  # normalization of vdws

        x = torch.from_numpy(x).float()
        y = torch.FloatTensor(f1[pdbid].attrs['affinity'].reshape(1, -1)).view(-1, 1)
        edge_attr = torch.from_numpy(f2[pdbid][:, :]).float()
        edge_ind = torch.from_numpy(f3[pdbid][:, :]).long()
        data = Data(x=x, edge_index=edge_ind, edge_attr=edge_attr, y=y)

        f1.close()
        f2.close()
        f3.close()
        return data