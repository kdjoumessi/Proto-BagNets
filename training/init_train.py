import torch
from torch.utils.data import DataLoader


from utils.func import exit_with_error
from modules.scheduler import ClippedCosineAnnealingLR, ScheduledWeightedSampler


def initialize_lr_scheduler(cfg, joint_optimizer, all_optimizer): 
    scheduler_strategy = cfg.solver.lr_scheduler

    if not scheduler_strategy:
        join_lr_scheduler, all_joint_lr_scheduler = None, None
    else:
        scheduler_args = cfg.scheduler_args[scheduler_strategy]
        all_joint_lr_scheduler = None
        if scheduler_strategy == 'cosine':
            join_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(joint_optimizer, **scheduler_args)
        elif scheduler_strategy == 'multiple_steps':
            join_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(joint_optimizer, **scheduler_args)
        elif scheduler_strategy == 'step_lr':
            join_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, **scheduler_args)
        elif scheduler_strategy == 'clipped_cosine':
            join_lr_scheduler = ClippedCosineAnnealingLR(joint_optimizer, **scheduler_args)
        else:
            raise NotImplementedError('Not implemented learning rate scheduler.')      

    return join_lr_scheduler, all_joint_lr_scheduler

############
def initialize_optimizer(cfg, model):
    optimizer_strategy = cfg.solver.optimizer 
    learning_rate, warm_lr = cfg.solver.learning_rate, cfg.solver.warm_solver.learning_rate
    weight_decay, warm_wd = cfg.solver.weight_decay, cfg.solver.warm_solver.weight_decay
    momentum, nesterov = cfg.solver.momentum, cfg.solver.nesterov

    #if cfg.train.network=='ProtoPNet':
    if optimizer_strategy == 'SGD':
        warm_optimizer_specs = [
            {'params': model.add_on_layers.parameters(), 'lr': warm_lr[0], 'weight_decay': warm_wd[0]}, # , 'momentum': momentum, 'nesterov': nesterov
            {'params': model.prototype_vectors, 'lr': warm_lr[1]}] # , 'momentum': momentum, 'nesterov': nesterov
        warm_optimizer = torch.optim.SGD(warm_optimizer_specs)
        
        joint_optimizer_specs = [{'params': model.features.parameters(), 'lr': learning_rate[0], 
                                    'weight_decay': weight_decay[0]}, # , 'momentum': momentum, 'nesterov': nesterov
                                 {'params': model.add_on_layers.parameters(), 'lr': learning_rate[1],
                                    'weight_decay': weight_decay[1]}, # , 'momentum': momentum, 'nesterov': nesterov
                                 {'params': model.prototype_vectors, 'lr': learning_rate[2]} # 
                                 ]
        joint_optimizer = torch.optim.SGD(joint_optimizer_specs)

        last_layer_optimizer_specs = [
            {'params': model.last_layer.parameters() if cfg.train.fc_classification_layer else model.last_layer, 'lr': learning_rate[3]}] # , 'nesterov': nesterov, 'momentum': momentum
        last_layer_optimizer = torch.optim.SGD(last_layer_optimizer_specs)

        # all parameters
        all_layer_optimizer = None
    else:
        raise NotImplementedError('Not implemented optimizer.')
    
    return warm_optimizer, joint_optimizer, last_layer_optimizer, all_layer_optimizer

def initialize_dataloader(cfg, train_dataset, push_dataset, val_dataset):
    batch_size = cfg.train.batch_size
    num_workers = cfg.train.num_workers
    pin_memory = cfg.train.pin_memory
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=None,
        sampler=None,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory
    )
    push_loader = DataLoader(
        push_dataset,
        batch_size=batch_size,
        shuffle=None,
        sampler=None,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle= None,
        sampler=None,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory
    )

    return train_loader, push_loader, val_loader