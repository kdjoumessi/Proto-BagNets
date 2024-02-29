import os
import pandas as pd
from torch.utils.data import Dataset
import torch
from PIL import Image

class OCTDataset(Dataset):
    def __init__(self, cfg, train=True, test=False, transform=None):
        self.img_size = cfg.data.input_size
        self.transform = transform
        self.image_path = os.path.join(cfg.paths.root, cfg.paths.dset_dir)
        
        self.n_classes = cfg.data.num_classes
        self.str_label = cfg.data.target_col_name

        if not test: 
            if train:
                self.df = pd.read_csv(os.path.join(self.image_path, cfg.paths.train_csv))
            else:
                self.df = pd.read_csv(os.path.join(self.image_path, cfg.paths.val_csv))
        else:
            self.df = pd.read_csv(os.path.join(self.image_path, cfg.paths.test_csv))
            
        self.filenames = self.df['filenames']
        self.labels = self.df[self.str_label]
        self.classes = sorted(list(set(self.targets)))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = os.path.join(self.image_path, self.filenames[idx])        
        image = Image.open(filename) # generate PIL image

        if self.transform:
            image = self.transform(image)
            
        label = self.labels.iloc[idx]
        return image, label
    
    def balanced_weights(self):
        weights = [0] * len(self)

        for idx, val in enumerate(getattr(self.df, self.str_label)):
                weights[idx] = 1 / self.class_proportions[val]
        return weights

    @property
    def class_proportions(self):
        y = self.targets.view(-1, 1)
        targets_onehot = (y == torch.arange(self.n_classes).reshape(1, self.n_classes)).float()
        proportions = torch.div(torch.sum(targets_onehot, dim=0) , targets_onehot.shape[0])
        return proportions

    @property
    def targets(self):
        return torch.tensor(self.df[self.str_label].values)