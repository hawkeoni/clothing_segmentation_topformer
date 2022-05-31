from argparse import ArgumentParser

import torch
from mmcv.utils import Config
from net import Topformer, SimpleHead, TopformerSegmenter



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
    opt = optim.Adam(model.parameters())
    x = torch.rand(8, 3, 512, 512).to(device)
    print(model(x).shape)
    input()




if __name__ == "__main__":
    parser = ArgumentParser()
    args = parser.parse_args()
    main(args)