import numpy as np
import openbabel.pybel
import pandas as pd
import h5py
import xml.etree.ElementTree as ET
from prepare_utils import get_atoms_within_cutoff, get_edge_inter, get_edge_bond
from tfbio.data import Featurizer
import yaml
import sys

def create_hdf(affinity_data_path, output_node_hdf, output_edge_i_hdf, output_edge_a_hdf, general_PDBs_path,
               refined_PDBs_path, path_to_elements_xml, bad_pdbids_input=[]):
    """
    This function converts the mol2 files into one hdf file containing all complexes provided.
    input:
    1) path/to/cleaned/affinity/data.csv
    2) path/to/output/node/file.hdf
    3) path/to/output/edge_indices/file.hdf
    4) path/to/output/edge_attributes/file.hdf
    5) path/to/mol2/files
    6) path/to/PDBs/in/general_set
    7) path/to/PDBs/in/refined_set
    8) path/to/elements.xml
    9) bad_pdbids_input, an array containing any pdbids that crashed chimera or crashed this function. Set to [] by def
    ault
 
    output:
    1)  a complete hdf file containing featurized data for all of the PDB id's that will be used, saved as:
        'path/to/output/hdf/file.hdf'
    """

    # define function to select pocket mol2 files with atoms that have unrealistic charges
    def high_charge(molecule):
        for i, atom in enumerate(molecule):
            if atom.atomicnum > 1:
                if abs(atom.__getattribute__('partialcharge'))>2:
                    return True
                else: 
                    return False

    # define function to extract information from elements.xml file
    def parse_element_description(desc_file):
        element_info_dict = {}
        element_info_xml = ET.parse(desc_file)
        for element in element_info_xml.iter():
            if "comment" in element.attrib.keys():
                continue
            else:
                element_info_dict[int(element.attrib["number"])] = element.attrib

        return element_info_dict

    # define function to create a list of van der Waals radii for a molecule
    def parse_mol_vdw(mol, element_dict):
        vdw_list = []
        for atom in mol.atoms:
            if int(atom.atomicnum)>=2:
                vdw_list.append(float(element_dict[atom.atomicnum]["vdWRadius"]))
        return np.asarray(vdw_list)

    # read in data and format properly
    element_dict = parse_element_description(path_to_elements_xml)
    affinities = pd.read_csv(affinity_data_path)
    pdbids_cleaned = affinities['pdbid'].to_numpy()
    sets = affinities['set'].to_numpy()
    bad_complexes = bad_pdbids_input
    print(bad_complexes)

    affinities_ind = affinities.set_index('pdbid')['-logKd/Ki']
    featurizer = Featurizer()

    # define the cutoff
    cutoff = 6

    for i in range(len(pdbids_cleaned)):
        pdbid = pdbids_cleaned[i]
        setsplit = sets[i]

        if pdbid in bad_complexes:
            continue

        print('pdbid: %s, set: %s' % (pdbid, setsplit))

        # check whether pdbid exists or not
        f1 = h5py.File(output_node_hdf, 'a')
        f2 = h5py.File(output_edge_i_hdf, 'a')
        f3 = h5py.File(output_edge_a_hdf, 'a')

        if pdbid in f1.keys() and pdbid in f2.keys() and pdbid in f3.keys():
            continue

        # get names of pocket and ligand files
        if setsplit == 'general':
            pfile = general_PDBs_path + "/" + pdbid + '/' + pdbids_cleaned[i] + '_pocket.mol2'
            lfile = general_PDBs_path + "/" + pdbids_cleaned[i] + '/' + pdbids_cleaned[i] + '_ligand.mol2'
        else:
            pfile = refined_PDBs_path + "/" + pdbids_cleaned[i] + '/' + pdbids_cleaned[i] + '_pocket.mol2'
            lfile = refined_PDBs_path + "/" + pdbids_cleaned[i] + '/' + pdbids_cleaned[i] + '_ligand.mol2'

        print('pfile: %s' % pfile)
        print('lfile: %s' % lfile)

        # extract features from ligand and check for unrealistic charges
        ligand = next(openbabel.pybel.readfile('mol2', lfile))
        if high_charge(ligand):
            bad_complexes.append(pdbid)
            print(bad_complexes)
            continue

        ligand_coords, ligand_features = featurizer.get_features(ligand, molcode=1)
        ligand_vdw = parse_mol_vdw(mol=ligand, element_dict=element_dict)

        # extract features from pocket and check for unrealistic charges
        pocket = next(openbabel.pybel.readfile('mol2', pfile))
        if high_charge(pocket):
            bad_complexes.append(pdbid)
            print(bad_complexes)
            continue

        pocket_coords, pocket_features = featurizer.get_features(pocket, molcode=-1)
        pocket_vdw = parse_mol_vdw(mol=pocket, element_dict=element_dict)

        # center the ligand and pocket coordinates
        centroid = ligand_coords.mean(axis=0)
        ligand_coords -= centroid
        pocket_coords -= centroid

        # del the pocket atoms out of cutoff
        pocket_coords, pocket_features, pocket_vdw = get_atoms_within_cutoff(cutoff, pocket_coords, ligand_coords,
                                                                             pocket_features, pocket_vdw)

        # concatenate van der Waals radii into one numpy array
        vdw_radii = np.concatenate((ligand_vdw, pocket_vdw))
        vdw_radii = np.expand_dims(vdw_radii, 1)  # expand the dimension [[vdw1],[vdw2],[vdw3],...]

        # create the h5py file containing node embeddings and affinities
        with h5py.File(output_node_hdf, 'a') as f1:

            # delete existing dataset with the same pdbid
            if pdbid in f1.keys():
                del f1[pdbid]

            # assemble the features into one large numpy array: rows are heavy atoms, columns are coordinates and
            # features
            data_node = np.concatenate(
                (np.concatenate((ligand_coords, pocket_coords)),
                np.concatenate((ligand_features, pocket_features)), vdw_radii),
                axis=1
            )

            # create a new dataset for this complex in the hdf file
            dset_node = f1.create_dataset(pdbid, data=data_node, shape=data_node.shape,
                                        dtype='float32', compression='lzf')

            # add the affinity and van der Waals radii as attributes for this dataset
            dset_node.attrs['affinity'] = affinities_ind.loc[pdbid]

        # get indices and attributes of edges representing intermolecular interactions and bonds
        edge_inter_ind, edge_inter_attr = get_edge_inter(pocket_coords, ligand_coords, cutoff)
        l_edge_bond_ind, l_edge_bond_attr = get_edge_bond(ligand, ligand_features)
        p_edge_bond_ind, p_edge_bond_attr = get_edge_bond(pocket, pocket_features)

        # correct indices of pocket atoms
        p_edge_bond_ind = p_edge_bond_ind + len(ligand_features)

        edge_ind = np.concatenate((np.concatenate((edge_inter_ind, l_edge_bond_ind), axis=1), p_edge_bond_ind), axis=1)
        edge_attr = np.concatenate((np.concatenate((edge_inter_attr, l_edge_bond_attr), axis=0), p_edge_bond_attr),
                                   axis=0)

        # create the h5py file containing edge indices
        with h5py.File(output_edge_i_hdf, 'a') as f2:
            # delete existing dataset with the same pdbid
            if pdbid in f2.keys():
                del f2[pdbid]

            dset_edge_ind = f2.create_dataset(pdbid, data=edge_ind, shape=edge_ind.shape,
                                        dtype='int', compression='lzf')

        # create the h5py file containing edge attributes
        with h5py.File(output_edge_a_hdf, 'a') as f3:
            # delete existing dataset with the same pdbid
            if pdbid in f3.keys():
                del f3[pdbid]
            dset_edge_attr = f3.create_dataset(pdbid, data=edge_attr, shape=edge_attr.shape,
                                        dtype='int', compression='lzf')

    print(bad_complexes)


configfile = sys.argv[1]
with open(configfile, 'r', encoding='utf-8') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)

affinity_data_path = config['create_hdf']['affinity_data_path']
output_node_hdf = config['create_hdf']['output_node_hdf']
output_edge_i_hdf = config['create_hdf']['output_edge_i_hdf']
output_edge_a_hdf = config['create_hdf']['output_edge_a_hdf']
general_PDBs_path = config['create_hdf']['general_PDBs_path']
refined_PDBs_path = config['create_hdf']['refined_PDBs_path']
path_to_elements_xml = config['create_hdf']['path_to_elements_xml']

create_hdf(affinity_data_path, output_node_hdf, output_edge_i_hdf, output_edge_a_hdf, general_PDBs_path,
           refined_PDBs_path, path_to_elements_xml, bad_pdbids_input=[])
