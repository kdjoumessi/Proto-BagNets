from torch.utils.data import DataLoader

from .dataset import OCTDataset
from .transforms import oct_data_transforms


def generate_dataset(cfg, loader=False): 
    
    train_transform, train_push_transform, test_transform = oct_data_transforms(cfg)
    datasets = generate_OCT_dataset( 
        cfg,
        train_transform,
        train_push_transform,
        test_transform
    )

    if loader:
        train_dset, push_dset, val_dset, test_dset = datasets
        train_loader, push_loader, val_loader = initialize_dataloader(cfg, train_dset, push_dset, val_dset)
        return train_loader, push_loader, val_loader
    else:
        return datasets

def generate_OCT_dataset(cfg, train_transform, push_transform, test_transform):                
    dset_train = OCTDataset(cfg, transform=train_transform)
    dset_push_train = OCTDataset(cfg, transform=push_transform)
    dset_val = OCTDataset(cfg, train=False, transform=test_transform)
    dset_test = OCTDataset(cfg, train=False, test=True, transform=test_transform)
    return dset_train, dset_push_train, dset_test, dset_val


# define data loader
def initialize_dataloader(cfg, train_dataset, push_dset, val_dataset, weighted_sampler):
    batch_size = cfg.train.batch_size
    num_workers = cfg.train.num_workers
    pin_memory = cfg.train.pin_memory
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(weighted_sampler is None),
        sampler=weighted_sampler,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory
    )

    push_loader = DataLoader(
        push_dset,
        batch_size=batch_size,
        shuffle=(weighted_sampler is None),
        sampler=weighted_sampler,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory
    )

    return train_loader, push_loader, val_loader

def initialize_test_dataloader(cfg, dataset):
    batch_size = cfg.train.batch_size
    num_workers = cfg.train.num_workers
    pin_memory = cfg.train.pin_memory

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory
    )
    return loader