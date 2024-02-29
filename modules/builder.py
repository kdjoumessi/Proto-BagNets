import torch.nn as nn
from .base_architecture import BUILDER, base_architecture_to_features
from .protopnet import construct_PPNet

def generate_model(cfg):
    out_features = cfg.data.num_classes 
    model = build_model(cfg, out_features)
    model.to(cfg.base.device)

    multi_model = nn.DataParallel(model)
    multi_model.to(cfg.base.device)

    return model, multi_model

def build_model(cfg, output_nber):
    num_classes = cfg.data.num_classes
    pretrained = cfg.train.pretrained
    base_architecture = (cfg.train.base_architecture).lower()        
    model = construct_PPNet(cfg=cfg,
                            base_architecture=base_architecture,
                            pretrained=pretrained,
                            img_size=cfg.data.input_size,
                            prototype_shape=tuple(cfg.prototype.shape),
                            num_classes=num_classes,
                            add_on_layers_type=cfg.prototype.add_on_layers_type,
                            prototype_activation_function=cfg.prototype.activation_function,
                            backbone=base_architecture_to_features)
    return model