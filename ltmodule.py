import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset, Subset
#from torchvision.datasets import MNIST
import os
#from torchvision import datasets, transforms

from pytorch_lightning.core.lightning import LightningModule

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import net as net

import importlib as imp

imp.reload(net)

use_cuda = True




'''
have a separate load_data function so that we don't have to reload the data every time we make
changes to the LitData module
'''
#data, calendar, sell_prices, cat2val  = load_data()
def load_data():
    data = pd.read_csv('sales_train_validation.csv')
    calendar = pd.read_csv('calendar.csv')
    sell_prices = pd.read_csv('sell_prices.csv')
    print('finished reading csv data')

    calendar['wk'] = calendar['wm_yr_wk'].apply(lambda x: int(str(x)[-2:]))

    #make a numeric version of event_name and type_type
    for i in ['1', '2']:
        for cat in ['name', 'type']:
            unique_events = {k: v for v, k in enumerate(calendar['event_' + cat + '_' + i].unique())}
            calendar['event_' + cat + '_' + i + '_num'] = calendar['event_' + cat + '_' + i].apply(
                lambda x: unique_events[x])


    #cat2val maps non-numeric values to a number
    cat2val = {}

    for var in ['cat', 'dept', 'item', 'state', 'store']:
        cat2val[var] = {}
        for i, val in enumerate(sorted(data[var+'_id'].unique())):
            cat2val[var][val] = i


    sell_prices = sell_prices.set_index(['store_id', 'item_id']).sort_index()


    return data, calendar, sell_prices, cat2val




class LitData(LightningModule):

    def __init__(self, data, calendar, sell_prices, cat2val):
        super().__init__()

        #initialize data with pre-loaded csv files so we don't have to reload the data everytime LitData is updated
        self.cat2val = cat2val
        self.data = data
        self.calendar = calendar
        self.sell_prices = sell_prices

        self.hidden_dim = 16
        self.enc = self.getenc()
        self.dec = self.getdec()

        self.ds = Dataseq(self.data, self.calendar, self.sell_prices, self.cat2val, 'd_1', 'd_1885', 'd_1913')


        #split train-val by row in data.csv
        folds, fold_groups = self.train_val_split()
        fold = 0

        self.train_idx = np.array(list(itertools.chain.from_iterable([folds[i] for i in fold_groups[fold]['train']])))
        self.val_idx = np.array(list(itertools.chain.from_iterable([folds[i] for i in fold_groups[fold]['val']])))

        self.train_ds = Subset(self.ds, self.train_idx)
        self.val_ds = Subset(self.ds, self.val_idx)



    def train_val_split(self):
        data_idx = np.arange(self.data.shape[0])
        np.random.shuffle(data_idx)
        n_folds = 5
        fold_size = 1.0 * self.data.shape[0] / n_folds
        folds = [data_idx[int(i * fold_size):int((i + 1) * fold_size)] for i in range(6)]

        fold_groups = {}
        fold_groups[0] = {'train': [0, 1, 2, 3], 'val': [4]}
        fold_groups[1] = {'train': [1, 2, 3, 4], 'val': [0]}
        fold_groups[2] = {'train': [0, 2, 3, 4], 'val': [1]}
        fold_groups[3] = {'train': [0, 1, 3, 4], 'val': [2]}
        fold_groups[4] = {'train': [0, 1, 2, 4], 'val': [3]}

        return folds, fold_groups


    def getenc(self):
        #dummy_ds = Dataseq(self.data, self.calendar, self.sell_prices, self.cat2val, 'd_1', 'd_1885', 'd_1913')
        #dummy_enc, dummy_dec, dummy_y, dummy_w = dummy_ds.__getitem__(0)

        return net.Encoder(self.hidden_dim, self.cat2val)


    def getdec(self):
        #dummy_ds = Dataseq(self.data, self.calendar, self.sell_prices, self.cat2val, 'd_1', 'd_1885', 'd_1913')
        #dummy_enc, dummy_dec, dummy_y, dummy_w = dummy_ds.__getitem__(0)

        return net.Decoder(self.hidden_dim)


    def prepare_data(self):
        None


    def train_dataloader(self):
        kwargs = {'num_workers': 1, 'pin_memory': False} if use_cuda else {}
        #return DataLoader(self.train_ds, batch_size=64, shuffle=True, collate_fn=train_ds.collate_fn, **kwargs)
        return DataLoader(self.train_ds, batch_size=64, shuffle=True, **kwargs)

    def val_dataloader(self):
        kwargs = {'num_workers': 1, 'pin_memory': False} if use_cuda else {}
        #return DataLoader(self.train_ds, batch_size=64, shuffle=True, collate_fn=train_ds.collate_fn, **kwargs)
        return DataLoader(self.val_ds, batch_size=64, shuffle=False, **kwargs)



    def forward(self, X_enc, X_dec):
        h = self.enc(X_enc)
        pred = self.dec(X_dec, h)
        return pred


    def training_step(self, batch, batch_idx):
        X_enc, X_dec, y, w = batch
        pred = self(X_enc, X_dec)

        weighted_mse = torch.sum(w*torch.pow(pred - y, 2))/y.size(0)/y.size(1)

        return {'loss': weighted_mse}


    def validation_step(self, batch, batch_idx):
        X_enc, X_dec, y, w = batch
        pred = self(X_enc, X_dec)

        weighted_mse = torch.sum(w*torch.pow(pred - y, 2))/y.size(0)/y.size(1)

        return {'val_loss': weighted_mse}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()

        return {'val_loss': val_loss_mean}


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)






class Dataseq(Dataset):
    """dataset."""

    def __init__(self, data, calendar, sell_prices, cat2val, train_start_day, train_end_day, target_end_day):
        """
        Args:
            data: pandas dataframe
        """
        self.data = data
        self.calendar = calendar
        self.sell_prices = sell_prices
        self.cat2val = cat2val

        self.train_start_day = train_start_day
        self.train_end_day = train_end_day
        self.target_start_day = 'd_' + str(int(train_end_day.split('_')[1]) + 1)
        self.target_end_day = target_end_day

        self.calendar_features = ['wday', 'month', 'd', 'wk', 'event_name_1_num', 'event_type_1_num', 'event_name_2_num', 'event_type_2_num']
        self.ts_use = ['wday', 'month', 'event_name_1_num', 'event_type_1_num', 'event_name_2_num', 'event_type_2_num', 'wk', 'sell_price', 'sale_counts']

    def collate_fn(self, batch):
        X_enc = [item[0] for item in batch]
        X_dec = [item[1] for item in batch]
        y = [item[2] for item in batch]
        w = [item[3] for item in batch]
        return [X_enc, X_dec, y, w]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):


        #t = calendar[calendar_features + ['wm_yr_wk']]
        t = self.calendar[self.calendar_features + ['wm_yr_wk']]

        #store_id = data.loc[idx, 'store_id']
        #item_id = data.loc[idx, 'item_id']
        store_id = self.data.loc[idx, 'store_id']
        item_id = self.data.loc[idx, 'item_id']


        #this step is very slow
        #tsp = sell_prices[sell_prices.store_id == store_id][sell_prices.item_id == item_id][['wm_yr_wk', 'sell_price']]
        #tsp = self.sell_prices[self.sell_prices.store_id == store_id][self.sell_prices.item_id == item_id][['wm_yr_wk', 'sell_price']]
        #b = (sell_prices['store_id'] == store_id) & (sell_prices['item_id'] == item_id)
        #b = (self.sell_prices['store_id'] == store_id) & (self.sell_prices['item_id'] == item_id)

        #tsp = sell_prices[b]
        #tsp = self.sell_prices[b]
        tsp = self.sell_prices.loc[(store_id, item_id)]

        t = t.merge(tsp, how='left', left_on='wm_yr_wk', right_on='wm_yr_wk').fillna(-10.0)

        #t2 = data.loc[idx].T.iloc[6:].reset_index()
        #t2 = self.data.loc[idx].T.iloc[6:].reset_index()
        #t2.rename(columns={idx: 'sale_counts'}, inplace=True)


        #t = t.merge(t2, how='left', left_on='d', right_on='index')

        #this is faster than merge, but shadier... will not work if data has time steps missing...
        #t['sale_counts'] = t2['sale_counts']
        #print(self.data.loc[idx].T.iloc[6:])
        t['sale_counts'] = np.nan
        t.loc[0:self.data.shape[1]-7, 'sale_counts'] = self.data.loc[idx].T.iloc[6:].values
        t = t.set_index('d')



        X_enc = {k : v.values.astype('float64') for k, v in t.loc[self.train_start_day:self.train_end_day][self.ts_use].to_dict(orient='series').items()}
        #include the metadata
        for v in ['cat', 'dept', 'item', 'state', 'store']:
            X_enc[v] = self.cat2val[v][self.data.loc[idx, v+'_id']]


        X_dec = {k: v.values.astype('float64') for k, v in
             t.loc[self.target_start_day:self.target_end_day][self.ts_use].to_dict(orient='series').items()}



        return X_enc, {k:v for k, v in X_dec.items() if k != 'sale_counts'}, X_dec['sale_counts'], X_dec['sell_price']




