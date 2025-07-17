# Training in 256Hz data and 4s
import torch
from pytorch_lightning import loggers as pl_loggers

from engine_pretraining import *
from configs import *
torch.set_float32_matmul_precision("medium")

seed_torch(7)


# init model


model = LitEEGPT(get_config(**(MODELS_CONFIGS[tag])), 
                 USE_LOSS_A =(variant != "A"),
                 USE_LN     =(variant != "B"),
                 USE_SKIP   =(variant != "C"))
lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
callbacks = [lr_monitor]

trainer = pl.Trainer(strategy='auto', devices=devices, max_epochs=max_epochs, callbacks=callbacks,
                     logger=[pl_loggers.TensorBoardLogger('./logs/', name=f"EEGPT_{tag}_{variant}_tb"), 
                             pl_loggers.CSVLogger('./logs/', name=f"EEGPT_{tag}_{variant}_csv")])
trainer.fit(model, train_loader, valid_loader)