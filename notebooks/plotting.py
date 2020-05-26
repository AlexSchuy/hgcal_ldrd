# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import logging
import os
import os.path as osp
import shutil
import sys
import time
from pathlib import Path

import cycler
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
# %%
from sklearn.metrics import confusion_matrix
from tqdm.notebook import tqdm

# %%
INPUT_PATH = Path('../../data/single_tau/output/best_epoch')
files = sorted([str(p) for p in INPUT_PATH.glob('data*.pkl')])

# %%
data = pd.read_pickle(files[93])

# %%
class Event():
    @classmethod
    def from_file(cls, f):
        data = pd.read_pickle(f)
        X = data.loc[:, ['x', 'y', 'z', 'E', 't']]
        target = data.loc[:, 'target']
        prediction = data.loc[:, 'prediction'].astype(int)
        return cls(
            X = X, target = target,
            prediction = prediction
            )
    
    def __init__(self, X, target, prediction):
        self.X = X
        self.target = target
        self.prediction = prediction
        self.num_classes = 4
        self.num_hits = self.X.shape[0]
        
class EventFactory():
    def __init__(self, files, max_events=None):
        self.files = files
        self.max_events = max_events
        self.do_tqdm = False
        
    def __iter__(self):
        if self.do_tqdm:
            total = len(self.files)
            if not(self.max_events is None) and self.max_events < total:
                total = self.max_events
            iterator = tqdm(enumerate(self.files), total=total)
        else:
            iterator = enumerate(self.files)
        
        for i, f in iterator:
            if not(self.max_events is None) and i >= self.max_events: break
            yield Event.from_file(f)                


# %%
class PlotEventDisplay():
    def __init__(self, event):
        self.event = event

    def plot_output(self, label, title=None, savefig=None):
        fig, (ax0,ax1) = plt.subplots(1, 2, figsize=(20,12))        
        fig.set_facecolor('white')
        if title: fig.suptitle(title, fontsize=28)
        
        e_max = np.max(self.event.X.loc[:,'E'].values)
        e_normed = 100.*np.tanh(self.event.X.loc[:,'E'].values/e_max)
        
        color_cycler = cycler.cycler(color=['b', 'r', 'g', 'y', 'm', 'c'])()
        
        for i_cat in range(self.event.num_classes):
            color = next(color_cycler)['color']
            X = self.event.X.loc[label == i_cat]
            ax0.scatter(X.loc[:,'x'], X.loc[:,'z'], s=(e_normed), c=color)
            ax1.scatter(X.loc[:,'y'], X.loc[:,'z'], s=(e_normed), c=color)

        ax0.legend(['Noise', 'HAD', 'EM', 'MIP'], prop={'size': 24})
        fontsize = 24
        ax0.set_xlabel('x [cm]', fontsize=fontsize)
        ax0.set_ylabel('layer', fontsize=fontsize)
        ax1.set_xlabel('y [cm]', fontsize=fontsize)
        ax1.set_ylabel('layer', fontsize=fontsize)
        ax0.tick_params(axis='both', which='major', labelsize=20)
        ax1.tick_params(axis='both', which='major', labelsize=20)
        if savefig:
            fig.savefig(savefig)
            plt.close(fig)

    def plot_prediction(self, savefig=None):
        return self.plot_output(self.event.prediction, 'Prediction', savefig)
    
    def plot_truth(self, savefig=None):
        return self.plot_output(self.event.target, 'Truth', savefig)



# %%
if not osp.isdir('eventdisplays'): os.makedirs('eventdisplays')

backend_ =  matplotlib.get_backend() 
matplotlib.use("Agg")  # Prevent showing stuff

for i_event, event in enumerate(EventFactory(files, max_events=100)):
    plot = PlotEventDisplay(event)
    plot.plot_prediction('eventdisplays/{0:03d}_pred.png'.format(i_event))
    plot.plot_truth('eventdisplays/{0:03d}_truth.png'.format(i_event))

matplotlib.use(backend_) # Reset backend

# %%

class PlotStatistics():
    def __init__(self, files):
        self.files = files
        self.num_classes = 4
        # self.n_events = 100
        self.n_events = None
        self.conf_mat = None
        
    def events(self):
        return EventFactory(self.files, self.n_events)
    
    def average_confusion_matrix(self):
        target = np.concatenate([e.target.values for e in self.events()])
        prediction = np.concatenate([e.prediction.values for e in self.events()])
        conf_mat = confusion_matrix(target, prediction, labels=np.arange(self.num_classes), normalize='true').transpose()
        self.conf_mat = conf_mat
        return conf_mat

    def display_confusion_matrix(self):
        np.set_printoptions(suppress=True, precision=4)
        if self.conf_mat is None:
            self.average_confusion_matrix()
        print(self.conf_mat)
    
    def _energy_collection_perevent(self, event, i_cat):
        pred_energy = event.X.loc[(event.prediction == i_cat) & (event.X.loc[:, 'E'] >= 0.0), 'E'].sum()
        true_energy = event.X.loc[(event.target == i_cat) & (event.X.loc[:, 'E'] >= 0.0), 'E'].sum()
        return pred_energy/true_energy if true_energy > 0. else -1
    
    def get_energy_collection(self):
        energy_collection_per_cat = [ [] for i_cat in range(self.num_classes) ]
        for event in self.events():
            for i_cat in range(self.num_classes):
                energy_collection = self._energy_collection_perevent(event, i_cat)
                if energy_collection != -1:
                    energy_collection_per_cat[i_cat].append(energy_collection)
        return energy_collection_per_cat

    def plot_energy_collection_hist(self, energy_collection_per_cat=None, savefig=None):
        fig, axs = plt.subplots(2, 2, figsize=(12,12))
        color_cycler = cycler.cycler(color=['b', 'r', 'g', 'y', 'm', 'c'])()
        labels = [ 'Noise', 'HAD', 'EM', 'MIP' ]
        
        if energy_collection_per_cat is None:
            energy_collection_per_cat = self.get_energy_collection()
        
        for i_cat, ax in zip(range(self.num_classes), axs.ravel()):
            ax.hist(
                energy_collection_per_cat[i_cat],
                bins = np.linspace(0.5, 1.5, 50),
                histtype = u'step', label = labels[i_cat],
                **next(color_cycler)
                )
            ax.legend(fontsize=14)
            ax.set_ylabel('Count', fontsize=14)
            ax.set_xlabel('E_pred / E_true', fontsize=14)
            
        if savefig: fig.savefig(savefig, bbox_inches='tight')

plot = PlotStatistics(files)

# %%
print([sum(e.prediction.loc[e.prediction==i].shape[0] for e in EventFactory(files)) for i in range(4)])
print([sum(e.target.loc[e.target==i].shape[0] for e in EventFactory(files)) for i in range(4)])


# %%
plot.display_confusion_matrix()


# %%
# plot.n_events = 10
plot.n_events = None
energy_collection_per_cat = plot.get_energy_collection()


# %%
plot.plot_energy_collection_hist(energy_collection_per_cat, savefig='energycollection.pdf')


# %%
summaries = np.load('../output/summaries.npz')



# %%
runs = {k: summaries[k] for k in summaries.keys()}


# %%
def plot_valid_loss():
    fig, axs = plt.subplots(2, 2, figsize=(12,12))
    
    axs[0,0].plot(runs['epoch'], runs['valid_loss'], label='Validation loss')
    axs[0,0].plot(runs['epoch'], runs['train_loss'], label='Training loss')
    axs[0,0].set_xlabel('Epoch', fontsize=14)
    axs[0,0].set_ylabel('Loss', fontsize=14)
    axs[0,0].legend(fontsize=14)

    axs[0,1].plot(runs['epoch'], runs['valid_acc'])
    axs[0,1].set_xlabel('Epoch', fontsize=14)
    axs[0,1].set_ylabel('Validation accuracy', fontsize=14)

    axs[1,0].plot(runs['epoch'], runs['lr'])
    axs[1,0].set_xlabel('Epoch', fontsize=14)
    axs[1,0].set_ylabel('Learning rate', fontsize=14)

    axs[1,1].plot(runs['epoch'], np.array(runs['train_time']) / 60.)
    axs[1,1].set_xlabel('Epoch', fontsize=14)
    axs[1,1].set_ylabel('Training time [min]', fontsize=14)

plot_valid_loss()

# %%
for i in range(4):
    target = np.concatenate([e.target.values for e in EventFactory(files)])
    
    energy = np.concatenate([e.X.loc[e.target==i, 'E'].values for e in EventFactory(files)])
    print(np.sum(energy < 0.0) / energy.shape[0])
# %%
