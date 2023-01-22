import torch as ch
import torch.nn as nn
from torchvision import models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchmetrics
import numpy as np
import itertools
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule

import os

from collections import OrderedDict


def k_from_size(xshape, size):
    shape = (xshape[1], xshape[2]*xshape[3])
    k = np.round(size / (2*shape[0] + shape[1]))
    return k

def size_from_k(xshape, k):
    shape = (xshape[1], xshape[2]*xshape[3])
    return k*(2*shape[0] + shape[1])

class SketchLayer(nn.Module):
    def __init__(self, c, k):
        super(SketchLayer, self).__init__()
        s1 = nn.Linear(c,k, bias=False)
        s2 = nn.Linear(k,c, bias=False)
        self.sketch_layer = nn.Sequential(s1, s2)

    def forward(self, x):
        shape = x.shape
        x = ch.reshape(x,(x.shape[0], x.shape[1], x.shape[2]*x.shape[3]))
        xhat = self.sketch_layer(x.transpose(1,2)) 
        xhat = ch.reshape(xhat.transpose(1,2), shape)
        return xhat


class SketchModel(LightningModule):
    def __init__(self, model, sync_dist=True, optimizer_params = {'optimizer':'sgd', 'lr': 0.01, 'lr_scheduler': None, 'lr_scheduler_args': {}}, track_grads=None):
        super().__init__()
        self.model = model
        self.train_top1_accuracy = torchmetrics.Accuracy()
        self.train_top5_accuracy = torchmetrics.Accuracy(top_k=5)
        self.val_top1_accuracy = torchmetrics.Accuracy()
        self.val_top5_accuracy = torchmetrics.Accuracy(top_k=5)
        self.loss = ch.nn.CrossEntropyLoss()
        self.sync_dist = sync_dist
        self.optimizer_params = optimizer_params
        self.save_hyperparameters(ignore=["model"])
        self.track_grads=track_grads

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, target = batch
        output = self.forward(images)
        loss = self.loss(output, target)
        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
        self.log('train_top1_acc', self.train_top1_accuracy(output, target), sync_dist = self.sync_dist, on_epoch=True, on_step=False, prog_bar=True)
        self.log('train_top5_acc', self.train_top5_accuracy(output, target), sync_dist = self.sync_dist, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, target = batch
        output = self.forward(images)
        loss = self.loss(output, target)
        self.log("val_loss", loss, on_epoch=True, on_step=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
        self.log('val_top1_acc', self.val_top1_accuracy(output, target), sync_dist = self.sync_dist, on_epoch=True, on_step=True, prog_bar=True)
        self.log('val_top5_acc', self.val_top5_accuracy(output, target), sync_dist = self.sync_dist, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)


    def configure_optimizers(self):
    
        if self.optimizer_params['optimizer'] == 'sgd':
            optimizer = ch.optim.SGD(self.model.parameters(), lr = self.optimizer_params['lr'], momentum = 0.9)
        elif self.optimizer_params['optimizer'] == 'adam':
            optimizer = ch.optim.Adam(self.model.parameters(), self.optimizer_params['lr'])
        elif self.optimizer_params['optimizer'] == 'adagrad':
            optimizer = ch.optim.Adagrad(self.model.parameters(), self.optimizer_params['lr'])
        else:
            raise NotImplementedError
        
        if not self.optimizer_params['lr_scheduler']:
            return optimizer
        
        elif self.optimizer_params['lr_scheduler'] == 'reduce_on_plateau':
            scheduler = ch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                mode=self.optimizer_params['lr_scheduler_args']['mode'], 
                                                                factor=self.optimizer_params['lr_scheduler_args']['factor'], 
                                                                patience=self.optimizer_params['lr_scheduler_args']['patience'], 
                                                                threshold=self.optimizer_params['lr_scheduler_args']['threshold'], 
                                                                threshold_mode=self.optimizer_params['lr_scheduler_args']['threshold_mode'], 
                                                                min_lr=self.optimizer_params['lr_scheduler_args']['min_lr'],
                                                                verbose=True)

            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_top1_acc_epoch"}
        
        elif self.optimizer_params['lr_scheduler'] == 'steplr':
            scheduler = ch.optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size=self.optimizer_params['lr_scheduler_args']['step_size'], 
                                                    gamma=self.optimizer_params['lr_scheduler_args']['gamma'], 
                                                    verbose=True)
            return [optimizer], [scheduler]
        else:
            raise NotImplementedError
       
    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        # example to inspect gradient information in tensorboard
        if self.track_grads:
            if self.trainer.global_step % self.track_grads == 0:  # don't make the tf file huge
                for k, v in self.model.named_parameters():
                    if v.requires_grad:
                        self.logger.experiment.add_histogram(
                            tag=k, values=v.grad, global_step=self.trainer.global_step
                        )
        

def insert_sketch_into_resnet(model, section, section_idx, sketch_dict, freeze_model = True, unfreeze_all_sketches = False, sketch_init = 'default'):
    
    for layer_name in sketch_dict:
        S = SketchLayer(sketch_dict[layer_name][0], sketch_dict[layer_name][1])
        model._modules[section][section_idx]._modules[layer_name+'input_sketch'] = S
        old_layer = model._modules[section][section_idx]._modules[layer_name]
        new_layer = nn.Sequential(model._modules[section][section_idx]._modules[layer_name+'input_sketch'],
                                  old_layer)
        model._modules[section][section_idx]._modules[layer_name] = new_layer
        # add first sketch layer to skip connection feature map as well
        if layer_name=='conv1':
            if 'downsample' in model._modules[section][section_idx]._modules:
                old_downsample = model._modules[section][section_idx]._modules['downsample']
                new_downsample = nn.Sequential(model._modules[section][section_idx]._modules[
                    layer_name+'input_sketch'], old_downsample)
                model._modules[section][section_idx]._modules['downsample'] = new_downsample 
            else:
                model._modules[section][section_idx]._modules['downsample'] = model._modules[
                    section][section_idx]._modules[layer_name+'input_sketch']
                model._modules[section][section_idx].downsample = model._modules[
                    section][section_idx]._modules[layer_name+'input_sketch']
                
    if freeze_model:
        for param in model.parameters():
            param.requires_grad = False
        for layer_name in sketch_dict:
            for m in model._modules[section][section_idx]._modules[layer_name].modules():
                if isinstance(m, SketchLayer):
                    for param in m.parameters():
                        param.requires_grad = True

    if unfreeze_all_sketches:
        for m in model.modules():
            if isinstance(m, SketchLayer):
                for param in m.parameters():
                    param.requires_grad = True
    
    return model

def insert_multiple_sketches_into_resnet(model,  
                                    sketch_dicts, 
                                    freeze_model = True, 
                                    unfreeze_all_sketches = False, 
                                    sketch_init = 'default', 
                                    sketch_type='post_act'):
    
    for (section, section_dicts) in sketch_dicts.items():
        for (section_idx, section_idx_dict) in section_dicts:
            model = insert_sketch_into_resnet(model, section, section_idx, section_idx_dict, freeze_model = freeze_model, unfreeze_all_sketches = unfreeze_all_sketches, sketch_init=sketch_init)
        
    return model

def unfreeze_sketches(model, sketch_idxs):
    count = 1
    for m in model.modules():
        if isinstance(m, SketchLayer):
            if count in sketch_idxs:
                for param in m.parameters():
                    param.requires_grad = True
            else:
                for param in m.parameters():
                    param.requires_grad = False
            count+=1
    return model

def construct_resnet_class_for_model_load(existing_sketch_dicts,  
                                    sketch_type='post_act', 
                                    freeze_model = True, 
                                    unfreeze_all_sketches = False,
                                    arch='resnet50'):
    if arch=='resnet50':
        model = models.resnet50(pretrained=True)
    elif arch=='resnet18':
        model = models.resnet18(pretrained=True)
    else:
        raise NotImplementedError
    model = insert_multiple_sketches_into_resnet(model, 
                                    sketch_dicts=existing_sketch_dicts,  
                                    sketch_type=sketch_type,
                                    freeze_model = freeze_model, 
                                    unfreeze_all_sketches = unfreeze_all_sketches)

    return model


def insert_sketch_into_vgg(model, section, section_idx, sketch_dict,
                           freeze_model = True, unfreeze_all_sketches = False):

    for layer in sketch_dict:
        S = SketchLayer(sketch_dict[layer][0], sketch_dict[layer][1])
        old_layer = model._modules[section][section_idx]
        new_layer = nn.Sequential(S,old_layer)
        model._modules[section][section_idx] = new_layer
        
    if freeze_model:
        for param in model.parameters():
            param.requires_grad = False
        for m in model._modules[section][section_idx].modules():
            if isinstance(m, SketchLayer):
                for param in m.parameters():
                    param.requires_grad = True
    
    if unfreeze_all_sketches:
        for m in model.modules():
            if isinstance(m, SketchLayer):
                for param in m.parameters():
                    param.requires_grad = True
    
    return model


def insert_multiple_sketches_into_vgg(model,  
                                    sketch_dicts, 
                                    freeze_model = True, 
                                    unfreeze_all_sketches = False, 
                                    sketch_init = 'default'):
    
    for ((section, section_idx), sketch_dict) in sketch_dicts.items():
        model = insert_sketch_into_vgg(model, section, section_idx, sketch_dict,
                                       freeze_model = freeze_model,
                                       unfreeze_all_sketches = unfreeze_all_sketches)
        
    return model

def construct_vgg_class_for_model_load(existing_sketch_dicts, arch='vgg19'):
    if arch == 'vgg19':
        model = models.vgg19_bn(pretrained=True)
    elif arch == 'vgg16':
        model = models.vgg16_bn(pretrained=True)
    else:
        raise NotImplementedError

    model = insert_multiple_sketches_into_vgg(model,
                                              sketch_dicts=existing_sketch_dicts)
    return model


def get_imagenet_dataloaders(imagenet_path, num_workers, batch_size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dir = os.path.join(imagenet_path, "train")
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    # all stages will use the eval dataset
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_dir = os.path.join(imagenet_path, "val")
    eval_dataset = datasets.ImageFolder(
                                        val_dir,
                                        transforms.Compose(
                                            [
                                                transforms.Resize(256),
                                                transforms.CenterCrop(224), 
                                                transforms.ToTensor(), 
                                                normalize
                                            ]
                                        ),
    )

    train_loader = ch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
            )

    val_loader = ch.utils.data.DataLoader(
                dataset=eval_dataset, 
                batch_size=batch_size, 
                num_workers=num_workers, 
                pin_memory=True
            )  

    return train_loader, val_loader 

