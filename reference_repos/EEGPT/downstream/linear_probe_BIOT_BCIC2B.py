import random 
import os
import torch
from torch import nn
import pytorch_lightning as pl

from functools import partial
import numpy as np
import random
import os 
import tqdm
from pytorch_lightning import loggers as pl_loggers
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

from Modules.Network.utils import Conv1dWithConstraint, LinearWithConstraint

from Modules.BIOT.biot import (
    BIOTClassifier,
)
import torch
from utils_eval import get_metrics




class LitEEGPTCausal(pl.LightningModule):

    def __init__(self, pretrain_model_choice = 0):
        super().__init__() 
        pretrain_models = ["Modules/BIOT/EEG-PREST-16-channels.ckpt",
                           "Modules/BIOT/EEG-SHHS+PREST-18-channels.ckpt",
                           "Modules/BIOT/EEG-six-datasets-18-channels.ckpt"]
        if pretrain_model_choice == 0: in_channels = 16
        elif pretrain_model_choice == 1: in_channels = 18
        elif pretrain_model_choice == 2: in_channels = 18
        else: raise ValueError("pretrain_model_choice should be 0, 1, or 2")
        
        self.chan_conv      = Conv1dWithConstraint(3, in_channels, 1, max_norm=1)
        model = BIOTClassifier(
                    n_classes=4,
                    # set the n_channels according to the pretrained model if necessary
                    n_channels=in_channels,
                    n_fft=200,
                    hop_length=100,
                )
        model.biot.load_state_dict(torch.load(pretrain_models[pretrain_model_choice]))
        print(f"load pretrain model from {pretrain_models[pretrain_model_choice]}")
        for p in model.biot.parameters():
            p.requires_grad = False
        self.feature        = model
        self.loss_fn        = torch.nn.CrossEntropyLoss()
        self.running_scores = {"train":[], "valid":[], "test":[]}
        self.is_sanity=True
    
    def forward(self, x):
        B, C, T = x.shape
        if T%200!=0: 
            x = x[:,:,0:T-T%200]
            T = T-T%200
        x = self.chan_conv(x)
        pred = self.feature(x)
        return x, pred

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        y = F.one_hot(y.long(), num_classes=4).float()
        
        label = y
        
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        accuracy = ((torch.argmax(logit, dim=-1)==torch.argmax(label, dim=-1))*1.0).mean()
        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        self.log('train_acc', accuracy, on_epoch=True, on_step=False)
        self.log('data_avg', x.mean(), on_epoch=True, on_step=False)
        self.log('data_max', x.max(), on_epoch=True, on_step=False)
        self.log('data_min', x.min(), on_epoch=True, on_step=False)
        self.log('data_std', x.std(), on_epoch=True, on_step=False)
        
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
        
        metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "cohen_kappa", "f1", "roc_auc"]
        results = get_metrics(y_score.cpu().numpy(), label.cpu().numpy(), metrics, True)
        
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
        
        y_score =  logit
        y_score =  torch.softmax(y_score, dim=-1)[:,1]
        self.running_scores["valid"].append((label.clone().detach().cpu(), y_score.clone().detach().cpu()))

        return loss
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(
            list(self.chan_conv.parameters())+
            list(self.feature.parameters()),
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
        
# load configs
# -- LOSO 

# load configs
# -- LOSO 
from utils import *
import math
data_path = "../datasets/downstream/Data/BCIC_2b_0_38HZ/"
pretrain_model_choice=0
seed_torch(7)
for i in range(1,10):
    all_subjects = [i]
    all_datas = []
    train_dataset,valid_dataset,test_dataset = get_data(i,data_path,1, is_few_EA = True, target_sample=200*4)
    
    global max_epochs
    global steps_per_epoch
    global max_lr

    batch_size=64

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    
    max_epochs = 100
    steps_per_epoch = math.ceil(len(train_loader) )
    max_lr = 4e-4

    # init model
    model = LitEEGPTCausal(pretrain_model_choice)

    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    callbacks = [lr_monitor]
    
    trainer = pl.Trainer(accelerator='cuda',
                         max_epochs=max_epochs, 
                         callbacks=callbacks,
                         logger=[pl_loggers.TensorBoardLogger('./log/', name="BIOT_BCIC2B_tb", version="subject{}_model{}".format(i,pretrain_model_choice)), 
                                 pl_loggers.CSVLogger('./logs/', name="BIOT_BCIC2B_csv")])

    trainer.fit(model, train_loader, valid_loader, ckpt_path='last')