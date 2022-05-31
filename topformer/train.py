from argparse import ArgumentParser

import torch
from mmcv.utils import Config
from net import Topformer, SimpleHead, TopformerSegmenter



def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device", device)
    cfg = Config.fromfile("topformer/config.py")
    b_cfg = cfg.model.backbone
    backbone = Topformer(b_cfg.cfgs, b_cfg.channels, b_cfg.out_channels, b_cfg.embed_out_indice)
    h_cfg = cfg.model.decode_head

    head = SimpleHead(
        in_channels=h_cfg.in_channels,
        channels=h_cfg.channels,
        num_classes=h_cfg.num_classes,
        in_index=h_cfg.in_index,
    )
    model = TopformerSegmenter(backbone, head).to(device)
    x = torch.rand(1, 3, 512, 512).to(device)
    print(model(x).shape)




if __name__ == "__main__":
    parser = ArgumentParser()
    args = parser.parse_args()
    main(args)