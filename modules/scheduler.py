import torch
from torch.utils.data.sampler import Sampler

class WarmupLRScheduler():
    def __init__(self, optimizer, warmup_epochs, initial_lr, params):
        self.epoch = 0
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.params = params

    def step(self):
        if self.epoch <= self.warmup_epochs:
            self.epoch += 1
            curr_lr = (self.epoch / self.warmup_epochs) * self.initial_lr[1]

            for param_group in self.optimizer.param_groups[1:]:
                param_group['lr'] = curr_lr
            
            self.optimizer.param_groups[1]['weight_decay'] = self.params[0] # only for the 1x1 conv layers
            self.optimizer.param_groups[1]['momentum'] = self.params[1]
            self.optimizer.param_groups[1]['nesterov'] = self.params[2]

    def is_finish(self):
        return self.epoch >= self.warmup_epochs

class ClippedCosineAnnealingLR():
    def __init__(self, optimizer, T_max, min_lr):
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        self.min_lr = min_lr
        self.finish = False

    def step(self):
        if not self.finish:
            self.scheduler.step()
            curr_lr = self.optimizer.param_groups[0]['lr']
            if curr_lr < self.min_lr:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.min_lr
                self.finish = True

    def is_finish(self):
        return self.finish
    
class ScheduledWeightedSampler(Sampler):
    def __init__(self, dataset, decay_rate):
        self.dataset = dataset
        self.decay_rate = decay_rate

        self.num_samples = len(dataset)
        self.targets = [sample[1] for sample in dataset.imgs]
        self.class_weights = self.cal_class_weights()

        self.epoch = 0
        self.w0 = torch.as_tensor(self.class_weights, dtype=torch.double)
        self.wf = torch.as_tensor([1] * len(self.dataset.classes), dtype=torch.double)
        self.sample_weight = torch.zeros(self.num_samples, dtype=torch.double)
        for i, _class in enumerate(self.targets):
            self.sample_weight[i] = self.w0[_class]

    def step(self):
        if self.decay_rate < 1:
            self.epoch += 1
            factor = self.decay_rate**(self.epoch - 1)
            self.weights = factor * self.w0 + (1 - factor) * self.wf
            for i, _class in enumerate(self.targets):
                self.sample_weight[i] = self.weights[_class]

    def __iter__(self):
        return iter(torch.multinomial(self.sample_weight, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples

    def cal_class_weights(self):
        num_classes = len(self.dataset.classes)
        classes_idx = list(range(num_classes))
        class_count = [self.targets.count(i) for i in classes_idx]
        weights = [self.num_samples / class_count[i] for i in classes_idx]
        min_weight = min(weights)
        class_weights = [weights[i] / min_weight for i in classes_idx]
        return class_weights