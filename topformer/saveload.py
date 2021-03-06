import torch

from net import Topformer, TopformerSegmenter, SimpleHead
from scheduler import get_optimizer_and_scheduler



def save_ckpt(config, model, optimizer, scheduler, path):
    m = model.state_dict()
    o = optimizer.state_dict()
    state = {"config": config, "model": m, "optimizer": o}
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()

    torch.save(state, open(path, "wb"))


def load_model(path, device=torch.device("cuda"), return_cfg=False):
    ckpt = torch.load(path)
    cfg = ckpt["config"]
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
    model = TopformerSegmenter(backbone, head)
    model_state_dict = ckpt["model"]
    fixed_state_dict = {k[len("module."):]: v for k, v in model_state_dict.items()}
    model.load_state_dict(fixed_state_dict)
    model = model.to(device).eval()
    if return_cfg:
        return model, cfg
    return model
