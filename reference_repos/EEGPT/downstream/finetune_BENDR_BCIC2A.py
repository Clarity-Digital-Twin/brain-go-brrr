
import random 
import os
from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn
# torch.set_float32_matmul_precision('medium' )#| 'high'
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
import math
from functools import partial
import numpy as np
import random
import os 
import tqdm
# os.environ['CUDA_LAUNCH_BLOCKING']='1'
from pytorch_lightning import loggers as pl_loggers

from Modules.models.dn3_ext import BENDR, ConvEncoderBENDR

import torch.nn.functional as F
def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
seed_torch(7)


from utils import *
from utils_eval import get_metrics


use_channels_names=[
            'EEG-1', 'EEG-3',
    'EEG-0', 'EEG-1', 'EEG-Fz', 'EEG-3', 'EEG-4',
    'EEG-5', 'EEG-C3', 'EEG-Cz', 'EEG-C4', 'EEG-8',
    'EEG-9', 'EEG-10', 'EEG-Pz', 'EEG-12', 'EEG-13',
            'EEG-14', 'EEG-15'
    ]
ch_names = ['EEG-Fz', 'EEG-0', 'EEG-1', 'EEG-2', 'EEG-3', 'EEG-4', 'EEG-5', 'EEG-C3', 'EEG-6', 'EEG-Cz', 
                'EEG-7', 'EEG-C4', 'EEG-8', 'EEG-9', 'EEG-10', 'EEG-11', 'EEG-12', 'EEG-13', 'EEG-14', 'EEG-Pz', 
                'EEG-15', 'EEG-16', 'EOG-left', 'EOG-central', 'EOG-right']
# -- get channel id by use chan names
choice_channels = []
for ch in use_channels_names:
    choice_channels.append([x.lower().strip('.') for x in ch_names].index(ch.lower()))
use_channels = choice_channels
# use_channels = None

class LitBENDR(pl.LightningModule):

    def __init__(self):
        super().__init__()

        encoder = ConvEncoderBENDR(20, encoder_h=512, dropout=0., projection_head=False)
        encoder.load("Modules/models/encoder.pt")
        
        self.model = encoder
        self.scale_param    = torch.nn.Parameter(torch.tensor(1.))
        self.linear_probe   = torch.nn.Linear(5632, 4)
        
        self.drop           = torch.nn.Dropout(p=0.10)
        
        self.loss_fn        = torch.nn.CrossEntropyLoss()
        
    def mixup_data(self, x, y, alpha=None):
        # 随机选择另一个样本来混合数据
        
        lam = torch.rand(1).to(x) if alpha is None else alpha
        lam = torch.max(lam, 1 - lam)

        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_y = lam * y + (1 - lam) * y[index]

        return mixed_x, mixed_y
    
    def forward(self, x):
        x = torch.cat([x, self.scale_param.repeat((x.shape[0], 1, x.shape[-1]))], dim=-2)
        
        h = self.model(x)
        
        h = h.flatten(1)
        h = self.drop(h)
        
        pred = self.linear_probe(h)
        
        return x, pred

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        soft_labels = F.one_hot(y.long(), num_classes=4).float()
        x,y = self.mixup_data(x,soft_labels)
        
        label = y
        
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        accuracy = ((torch.argmax(logit, dim=-1)==torch.argmax(label, dim=-1))*1.0).mean()
        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('train_acc', accuracy, on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_avg', x.mean(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_max', x.max(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_min', x.min(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_std', x.std(), on_epoch=True, on_step=False, sync_dist=True)
        
        return loss
        
    def on_validation_epoch_start(self) -> None:
        self.running_scores["valid"]=[]
        return super().on_validation_epoch_start()
    def on_validation_epoch_end(self) -> None:
        if self.is_sanity:
            self.is_sanity=False
            return super().on_validation_epoch_end()
            
        label, y_score = [], []
        for x,y in self.running_scores["valid"]:
            label.append(x)
            y_score.append(y)
        label = torch.cat(label, dim=0)
        y_score = torch.cat(y_score, dim=0)
        print(label.shape, y_score.shape)
        
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted", "f1_macro", "f1_micro"]
        results = get_metrics(y_score.cpu().numpy(), label.cpu().numpy(), metrics, False)
        
        for key, value in results.items():
            self.log('valid_'+key, value, on_epoch=True, on_step=False, sync_dist=True)
        return super().on_validation_epoch_end()
    
    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        label = y.long()
        
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        accuracy = ((torch.argmax(logit, dim=-1)==label)*1.0).mean()
        # Logging to TensorBoard by default
        self.log('valid_loss', loss, on_epoch=True, on_step=False)
        self.log('valid_acc', accuracy, on_epoch=True, on_step=False)
        
        self.running_scores["valid"].append((label.clone().detach().cpu(), logit.clone().detach().cpu()))
        return loss
    
    def test_step(self, batch, *args, **kwargs):
        
        x, y = batch
        label = y.long()
        
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        accuracy = ((torch.argmax(logit, dim=-1)==label)*1.0).mean()
        # Logging to TensorBoard by default
        self.log('test_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('test_acc', accuracy, on_epoch=True, on_step=False, sync_dist=True)
        
        return loss
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(
            list([self.scale_param])+
            list(self.model.parameters())+
            list(self.linear_probe.parameters()),
            weight_decay=0.01)#
        
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=max_epochs, pct_start=0.2)
        lr_dict = {
            'scheduler': lr_scheduler, # The LR scheduler instance (required)
            # The unit of the scheduler's step size, could also be 'step'
            'interval': 'step',
            'frequency': 1, # The frequency of the scheduler
            'monitor': 'val_loss', # Metric for `ReduceLROnPlateau` to monitor
            'strict': True, # Whether to crash the training if `monitor` is not found
            'name': None, # Custom name for `LearningRateMonitor` to use
        }
      
        return (
            {'optimizer': optimizer, 'lr_scheduler': lr_dict},
        )


data_path = "../datasets/downstream/Data/BCIC_2a_0_38HZ"
# load configs
for sub in range(1,10):
    train_dataset,valid_dataset,test_dataset = get_data(sub,data_path,1,True, use_channels=use_channels)
        
    global max_epochs
    global steps_per_epoch
    global max_lr

    batch_size=64

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    
    max_epochs = 100
    steps_per_epoch = math.ceil(len(train_loader) )
    max_lr = 1e-5

    # init model
    model = LitBENDR()

    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    callbacks = [lr_monitor]
    
    trainer = pl.Trainer(accelerator='cuda',
                         max_epochs=max_epochs, 
                         callbacks=callbacks,
                         logger=[pl_loggers.TensorBoardLogger('./logs/', name="BENDR_BCIC2A_tb", version=f"subject{sub}"), 
                                 pl_loggers.CSVLogger('./logs/', name="BENDR_BCIC2A_csv")])

    trainer.fit(model, train_loader, test_loader, ckpt_path='last')


