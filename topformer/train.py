from argparse import ArgumentParser

import torch
from mmcv.utils import Config
from net import Topformer, SimpleHead, TopformerSegmenter

class PolyLrUpdaterHook(LrUpdaterHook):

    def __init__(self,
                 power: float = 1.,
                 min_lr: float = 0.,
                 **kwargs) -> None:
        self.power = power
        self.min_lr = min_lr
        super().__init__(**kwargs)

    def get_lr(self, runner: 'runner.BaseRunner', base_lr: float):
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters
        coeff = (1 - progress / max_progress)**self.power
        return (base_lr - self.min_lr) * coeff + self.min_lr


def make_optimizer(model):
    optimizer = AdamW

    return optimizer, scheduler


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device", device)
    cfg = Config.fromfile("topformer/config.py")
    b_cfg = cfg.model.backbone
    backbone = Topformer(
        cfgs=b_cfg.cfgs, 
        channels=b_cfg.channels, 
        out_channels=b_cfg.out_channels, 
        embed_out_indice=b_cfg.embed_out_indice,
        decode_out_indices=b_cfg.decode_out_indices,
        depths=b_cfg.depths,
        c2t_stride=b_cfg.c2t_stride,
        drop_path_rate=b_cfg.drop_path_rate,
        # norm_cfg=b_cfg.norm_cfg,
        init_cfg=b_cfg.init_cfg,
        num_heads=b_cfg.num_heads,
        )
    h_cfg = cfg.model.decode_head

    head = SimpleHead(
        in_channels=h_cfg.in_channels,
        channels=h_cfg.channels,
        num_classes=h_cfg.num_classes,
        in_index=h_cfg.in_index,
        dropout_ratio=h_cfg.dropout_ratio,
    )
    import torch.optim as optim
    model = TopformerSegmenter(backbone, head).to(device)
    optimizer = optim.AdamW(
        {}
    )
    # opt = optim.AdamW({model.backbone.p}, lr=0.0003, betas=(0.9, 0.999), weight_decay=0.01,
                    
    )
    optimizer = dict(_delete_=True, type='AdamW', lr=0.0003, betas=(0.9, 0.999), weight_decay=0.01,
                    paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                    'norm': dict(decay_mult=0.),
                                                    'head': dict(lr_mult=10.)
                                                    }))

    lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
    x = torch.rand(8, 3, 768, 768).to(device)
    print(model(x).shape)
    input()




if __name__ == "__main__":
    parser = ArgumentParser()
    args = parser.parse_args()
    main(args)