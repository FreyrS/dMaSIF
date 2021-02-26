import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

top_dir = Path('..')
experiment_names = ['TangentConv_search_3L_16dim_12A_FIXED_binet_c_restarted_epoch43',
'TangentConv_search_3L_16dim_12A_FIXED_binet_g_restarted_epoch38',
'TangentConv_search_1L_8dim_12A_FIXED_binet_gc_epoch34',
'TangentConv_search_3L_8dim_12A_FIXED_binet_gc_restarted_epoch49',
'TangentConv_search_1L_16dim_12A_FIXED_binet_gc_epoch45']

with open(top_dir/'surface_data/raw/protein_surfaces/testing_ppi.txt') as f:
    testing_list = f.read().splitlines()

pdb_list = testing_list

for experiment_name in experiment_names:
    print(experiment_name)
    desc_dir = top_dir/f'preds/{experiment_name}'
    all_roc_aucs = []
    all_preds = []
    all_labels = []
    for i, pdb_id in enumerate(pdb_list):
        pdb_id1 = pdb_id.split('_')[0]+'_'+pdb_id.split('_')[1]
        pdb_id2 = pdb_id.split('_')[0]+'_'+pdb_id.split('_')[2]
        if i%100==0:
            print(i,np.mean(all_roc_aucs))

        try:
            desc1 = np.load(desc_dir/f'{pdb_id1}_predfeatures.npy')[:,16:16+16]
            desc2 = np.load(desc_dir/f'{pdb_id2}_predfeatures.npy')[:,16:16+16]
            xyz1 = np.load(desc_dir/f'{pdb_id1}_predcoords.npy')
            xyz2 = np.load(desc_dir/f'{pdb_id2}_predcoords.npy')
        except FileNotFoundError:
            continue

        dists = cdist(xyz1,xyz2)<1.0
        if dists.sum()<1:
            continue

        iface_pos1 = dists.sum(1)>0
        iface_pos2 = dists.sum(0)>0

        pos_dists1 = dists[iface_pos1,:]
        pos_dists2 = dists[:,iface_pos2]

        desc_dists = np.matmul(desc1,desc2.T)
        #desc_dists = 1/cdist(desc1,desc2)

        pos_dists = desc_dists[dists].reshape(-1)
        pos_labels = np.ones_like(pos_dists)
        neg_dists1 = desc_dists[iface_pos1,:][pos_dists1==0].reshape(-1)
        neg_dists2 = desc_dists[:,iface_pos2][pos_dists2==0].reshape(-1)

        #neg_dists = np.concatenate([neg_dists1,neg_dists2],axis=0)
        neg_dists = neg_dists1
        neg_dists = np.random.choice(neg_dists,200,replace=False)
        neg_labels = np.zeros_like(neg_dists)

        preds = np.concatenate([pos_dists,neg_dists])
        labels = np.concatenate([pos_labels,neg_labels])

        roc_auc = roc_auc_score(labels,preds)
        all_roc_aucs.append(roc_auc)
        all_preds.extend(list(preds))
        all_labels.extend(list(labels))


    fpr, tpr, thresholds = roc_curve(all_labels,all_preds)
    np.save(f'roc_curves/{experiment_name}_fpr.npy',fpr)
    np.save(f'roc_curves/{experiment_name}_tpr.npy',tpr)
    np.save(f'roc_curves/{experiment_name}_all_labels.npy',all_labels)
    np.save(f'roc_curves/{experiment_name}_all_preds.npy',all_preds)

