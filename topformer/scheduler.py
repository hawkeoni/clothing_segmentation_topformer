from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler


class PolyScheduler(_LRScheduler):
    
    def __init__(self, optimizer, warmup_iters, warmup_ratio, power, min_lr, max_iters):
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.power = power
        self.min_lr = min_lr
        self.max_iters = max_iters
        super().__init__(optimizer)
        
    def get_lr(self):
        if self._step_count <= self.warmup_iters:
            partial = self._step_count / self.warmup_iters
            new_lrs = []
            for lr in self.base_lrs:
                start = lr * self.warmup_ratio
                cur_lr = start + partial * (lr - start)
                new_lrs.append(cur_lr)
            return new_lrs
        
        steps = self._step_count - self.warmup_iters
        tot_steps = self.max_iters - self.warmup_iters
        coef = (1 - steps / tot_steps) ** self.power
        lrs = self.base_lrs.copy()
        lrs = [lr * coef for lr in lrs]
        return lrs



def get_optimizer_and_scheduler(model, lr, weight_decay, warmup_iters, warmup_ratio, power, min_lr, max_iters):
    # optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    no_decay = []
    decay = []
    head = []

    for name, p in model.named_parameters():
        if "backbone" in name and ("bn.bias" in name or "bn.weight" in name):
            no_decay.append(p)
        elif "backbone" in name:
            decay.append(p)
        else:
            head.append(p)
        
    optimizer = AdamW(
        [
            {"params": decay},
            {"params": no_decay, "weight_decay": 0},
            {"params": head, "lr": lr * 10}
        ],
        lr=lr, weight_decay=weight_decay
    )
    # for name, p in model.backbone.named_parameters():
    #     if "bn.bias" in name or "bn.weight" in name:
    #         no_decay.append(p)
    #     else:
    #         decay.append(p)
        
    # optimizer = AdamW(
    #     [
    #         {"params": decay},
    #         {"params": no_decay, "weight_decay": 0},
    #         {"params": model.segmentation_head.parameters(), "lr": lr * 10}
    #     ],
    #     lr=lr, weight_decay=weight_decay
    # )
    scheduler = PolyScheduler(optimizer, warmup_iters, warmup_ratio, power, min_lr, max_iters)
    return optimizer, scheduler
