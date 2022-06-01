import os
from time import time
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from mmcv.utils import Config

from scheduler import get_optimizer_and_scheduler
from net import Topformer, SimpleHead, TopformerSegmenter
from data import AlignedDataset
from saveload import save_ckpt


def get_batch(loader):
    while True:
        for batch in loader:
            yield batch


def setup():
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(
        backend="nccl", 
        world_size=world_size, 
        rank=rank)


def main(args):
    MAX_ITERS = args.max_iters
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
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    model = TopformerSegmenter(backbone, head).to(rank)
    model = DDP(model, device_ids=[rank])
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
        sampler=DistributedSampler(dataset, rank=rank, drop_last=True),
        pin_memory=True
    )

    criterion = nn.CrossEntropyLoss()
    
    start_time = time()
    if rank == 0:
        print("Start training")
        writer = SummaryWriter()
    optimizer.zero_grad()
    for iteration, (image, label) in enumerate(get_batch(dataloader), start=1):
        image = image.to(rank)
        label = label.to(rank)
        prediction = model(image)
        prediction = F.upsample(prediction, (args.fine_width, args.fine_height), mode="bilinear") 
        loss = criterion(prediction, label)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_val = loss.item()
        optimizer.zero_grad()
        if rank == 0:
            mean_loss = dist.all_reduce(loss, dist.ReduceOp.SUM).item() / world_size
            writer.add_scalar("loss", mean_loss, iteration)
            writer.add_scalar("lr", scheduler.get_lr()[0], iteration)

        if iteration % args.log_interval == 0 and rank == 0:
            now = time()
            etime = (now - start_time) / 60
            print(f"Iteration: {iteration}, Loss: {loss_val}, Elapsed Time: {etime}")
        
        if iteration % args.checkpoint_interval and rank == 0:
            save_folder = Path("checkpoints")
            save_folder.mkdir(exist_ok=True)
            name = f"model_{iteration}.pth"
            save_ckpt(cfg, model, optimizer, scheduler, save_folder / name)




# 3 min - 10 iterations; 1 iter = 8 samples; 80 samples; 4 gpu - 320 samples
# 320 / 3 = 100 samples/ min
# 45000 / 100 = 450 min / epoch = 8 hrs
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-iters", type=int, default=160_000)
    parser.add_argument("--image-folder", type=str)
    parser.add_argument("--df-path", type=str)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--fine-width", type=int, default=768)
    parser.add_argument("--fine-height", type=int, default=768)
    parser.add_argument("--mean", type=float, default=0.5)
    parser.add_argument("--std", type=float, default=0.5)
    args = parser.parse_args()
    setup()
    main(args)
