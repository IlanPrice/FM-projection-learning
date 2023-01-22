import torch as ch
import torch.nn as nn
from torchvision import models
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
# from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.plugins import DDPPlugin

import os
import time

from collections import OrderedDict
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import yaml 
import json

import utils
from utils import insert_multiple_sketches_into_vgg, SketchModel, get_imagenet_dataloaders, construct_vgg_class_for_model_load

parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument(
        "-c",
        "--config",
        dest="config_file",
        help="experiment config file",
        metavar="FILE",
        required=True,
    )

args = parser.parse_args()

with open(args.config_file, "r") as stream:
    cfg = yaml.load(stream, Loader=yaml.FullLoader)

# Set seed
ch.manual_seed(cfg['seed'])

# Build dataloaders
train_loader, val_loader = get_imagenet_dataloaders(cfg['dataset']['imagenet_path'],
                                                    num_workers=cfg['train']['num_workers'],
                                                    batch_size=cfg['train']['batch_size'])


layer_ks = []
for i in cfg['model']['sketch_dicts']:
    for j in cfg['model']['sketch_dicts'][i]:
        layer_ks.append(cfg['model']['sketch_dicts'][i][j][1])

if cfg['model']['existing_sketch_dicts'] is not None:
    existing_fm_sketch_idxs = [i[1] for i in list(cfg['model']['existing_sketch_dicts'].keys())]
else:
    existing_fm_sketch_idxs = []

fm_sketch_idxs = [i[1] for i in list(cfg['model']['sketch_dicts'].keys())]

# If not loading in a pre-trained sketch model, initialise model as pretrained VGG_BN
if cfg['model']['model_path'] is None:
    if cfg['model']['arch']=='vgg19':
        m = models.vgg19_bn(pretrained=True)
    elif cfg['model']['arch']=='vgg16':
        m = models.vgg16_bn(pretrained=True)
    else:
        raise NotImplementedError
# Otherwise construct a class matching the model to be loaded, and then load the state dict
else:
    m = construct_vgg_class_for_model_load(
        existing_sketch_dicts=cfg['model']['existing_sketch_dicts'],
        arch=cfg['model']['arch'],
    )
    m = SketchModel(m)
    m.load_state_dict(ch.load(cfg['model']['model_path'])['state_dict'])
    m = m.model

# Create new model with sketch layers inserted after the specified layers
insert_multiple_sketches_into_vgg(m, 
                                sketch_dicts=cfg['model']['sketch_dicts'], 
                                freeze_model = cfg['model']['freeze_model'], 
                                unfreeze_all_sketches = cfg['model']['unfreeze_all_sketches'],
                                )

print("sketches initialised")

# Initialise pytorch lightning model    
model = SketchModel(m, sync_dist=True, optimizer_params = cfg['optimizer'])

del m

print("model built")

# Early stopping callback for when target accuracy reached
early_stopping = pl.callbacks.EarlyStopping(monitor="val_top1_acc_epoch", min_delta=0.0,
    patience = cfg['train']['epochs'],
    mode='max', strict=True,
    stopping_threshold=cfg['train']['target_accuracy'])
# Callback to save best model
checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=cfg['train']['save_dir'], 
                                                    filename = cfg['train']['model_filename'],
                                                    monitor="val_top1_acc_epoch", 
                                                    mode='max')
# Callback to track LR
lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
# Creat list of callbacks
if cfg['train']['save_model']:
    callbacks = [checkpoint_callback, lr_monitor,  early_stopping]
else:
    callbacks = [lr_monitor,  early_stopping]
# Define logger
log_name = f"{cfg['model']['arch']}_ceiling_existing_sketches_after_{existing_fm_sketch_idxs}"
"_sketch_added_after_{fm_sketch_idxs}_k_{layer_ks}"
tb_logger = pl_loggers.TensorBoardLogger(save_dir = cfg['train']['log_dir'],
                                         name = log_name, 
                                         version = cfg['train']['version']
                                         )
# Initialise PL trainer  
trainer = pl.Trainer(precision=16, 
                     accelerator = "gpu", 
                     devices = cfg['train']['num_gpus'], 
                     num_nodes = 1,
                     max_epochs = cfg['train']['epochs'], 
                     callbacks= callbacks,
                     enable_checkpointing=cfg['train']['save_model'],
                     check_val_every_n_epoch=1, 
                     logger = tb_logger,
                     strategy=DDPPlugin(find_unused_parameters=False),
                     # strategy=DDPStrategy(find_unused_parameters=False),
                     )

print("trainer instantiated")

# Save config with logs
os.makedirs(cfg['train']['log_dir']+log_name, exist_ok=True )
with open(cfg['train']['log_dir']+ log_name +"/config.yaml", 'w') as outfile:
    yaml.dump(cfg, outfile)

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)



