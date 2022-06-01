from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from mmcv.utils import Config

from scheduler import get_optimizer_and_scheduler
from net import Topformer, SimpleHead, TopformerSegmenter
from data import AlignedDataset



def main(args):
    MAX_ITERS = args.max_iters
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
    model = TopformerSegmenter(backbone, head).to(device)
    optimizer, scheduler = get_optimizer_and_scheduler(
        model, lr=0.0003, weight_decay=0.01,
        warmup_iters=1500,
        warmup_ratio=1e-6,
        power=1,
        min_lr=0,
        max_iters=MAX_ITERS
    )
    dataset = AlignedDataset()
    dataset.initialize(args)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        sampler=RandomSampler(dataset),
        pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    
    print("Start training")
    for iteration, (image, label) in enumerate(dataloader):
        optimizer.zero_grad()
        image = image.to(device)
        label = label.to(device)
        prediction = model(image)
        print(image.shape, label.shape)
        print(prediction.shape)
        prediction = F.upsample(prediction, (args.fine_width, args.fine_height), mode="bilinear") 
        loss = criterion(prediction, label)
        loss.backward()
        optimizer.step()
        scheduler.step()
        print(iteration, loss)




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-iters", type=int, default=10_000)
    parser.add_argument("--image-folder", type=str)
    parser.add_argument("--df-path", type=str)
    parser.add_argument("--fine-width", type=int, default=768)
    parser.add_argument("--fine-height", type=int, default=768)
    parser.add_argument("--mean", type=float, default=0.5)
    parser.add_argument("--std", type=float, default=0.5)
    args = parser.parse_args()
    main(args)
