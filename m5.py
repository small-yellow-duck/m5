import pandas as pd
import matplotlib.pyplot as plt

import importlib as imp


from pytorch_lightning.core.lightning import LightningModule

import pytorch_lightning as pl


import ltmodule



#data, calendar, sell_prices, cat2val = ltmodule.load_data()
def main(data, calendar, sell_prices, cat2val):
    '''
    data, calendar, sell_prices, cat2val = ltmodule.load_data()
    ld = ltmodule.LitData(data, calendar, sell_prices, cat2val)
    loader = ld.train_dataloader()
    X_enc, X_dec, y, w = next(iter(ld.train_dataloader()))
    t = ld(X_enc, X_dec)
    '''

    trainer = pl.Trainer(gpus=1)
    model = ltmodule.LitData(data, calendar, sell_prices, cat2val)
    trainer.fit(model)