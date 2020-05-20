# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os, sys, shutil, time, os.path as osp, logging, numpy as np
from pathlib import Path
import torch
import matplotlib.pyplot as plt, cycler
import matplotlib
from torch_geometric.data import DataLoader


# %%
HGCAL_LDRD_PATH = osp.abspath('../../hgcal_ldrd/src')
PVCNN_PATH = osp.abspath('../../pvcnn')
sys.path.append(HGCAL_LDRD_PATH)


# %%
HGCAL_SCRIPTS_PATH = osp.abspath('../../hgcal_ldrd')
sys.path.append(HGCAL_SCRIPTS_PATH)


# %%
from datasets.hitgraphs import HitGraphDataset
sys.path.append(PVCNN_PATH)


# %%
from scripts import pvcnn_script


# %%
script = pvcnn_script.TrainingScript(debug=False)


# %%
script.load_checkpoint = '../output/checkpoints/model_checkpoint_PVConvForHGCAL_2562244_9c8b11eb88_alexjschuy.best.pth.tar'
full_dataset, train_dataset, valid_dataset = script.get_full_dataset()
valid_loader = DataLoader(valid_dataset, batch_size=script.valid_batch_size, shuffle=False)
trainer = script.get_trainer()


# %%
output_dir = Path('../../data/single_tau/output')
if not output_dir.exists():
    output_dir.mkdir(parents=True)
for i, d in enumerate(valid_dataset):
    one_file_subset = torch.utils.data.Subset(valid_dataset,[i])
    one_file_loader = DataLoader(one_file_subset, batch_size=1, shuffle=False)
    predictions = trainer.predict(one_file_loader)
    input = one_file_subset[0].x.cpu().numpy()
    output = np.zeros((input.shape[0], input.shape[1]+1))
    output[:, :-1] = input
    output[:, -1] = predictions
    output_path = output_dir / f'data_{i}.npy'
    np.save(output_path, output)

# %%


