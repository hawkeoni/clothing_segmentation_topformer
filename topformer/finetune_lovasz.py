
import os
import warnings
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
from saveload import save_ckpt, load_model
from common_segmentation import calculate_iou, get_palette, IMaterialistDataset
import segmentation_models_pytorch as smp
import numpy as np

PALETTE = get_palette(4)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


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
    if rank == 0:
        print("Finish setup")


def main(args):
    if dist.get_rank() == 0:
        print(args)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    MAX_ITERS = args.max_iters
    model = load_model(args.init_from)
    model = DDP(model, device_ids=[rank])
    optimizer, _ = get_optimizer_and_scheduler(
        model, lr=args.lr, weight_decay=0.01,
        warmup_iters=args.warmup_iters,
        warmup_ratio=1e-6,
        power=1,
        min_lr=0,
        max_iters=MAX_ITERS,
        use_scheduler = False
    )
    if rank == 0:
        print("Finish model creation")
    train_dataset = IMaterialistDataset("../images", "train_85.csv")
    val_dataset = IMaterialistDataset("../images", "val_15.csv")
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        sampler=DistributedSampler(train_dataset, rank=rank, drop_last=True),
        pin_memory=True,
        num_workers=8,
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        sampler=DistributedSampler(val_dataset, rank=rank, drop_last=True),
        pin_memory=True,
        num_workers=8,
    )
    validation_generator = get_batch(val_dataloader)
    
    classes = ["background", "upper", "lower", "whole"]

    criterion = smp.losses.LovaszLoss("multiclass")
    
    start_time = time()
    if rank == 0:
        print("Start training")
        from string import ascii_lowercase
        import random
        writer = SummaryWriter(log_dir="lovasz", flush_secs=10, filename_suffix="".join(random.choices(ascii_lowercase, k=10)))
    optimizer.zero_grad()
    for iteration, (image, label) in enumerate(get_batch(train_dataloader), start=1):
        image = image.to(rank)
        label = label.to(rank)
        prediction = model(image)
        prediction = F.upsample(prediction, (args.fine_width, args.fine_height), mode="bilinear") 
        loss = criterion(prediction, label)
        loss.backward()
        optimizer.step()
        loss_val = loss.item()
        optimizer.zero_grad()
        if rank == 0:
            writer.add_scalar("loss/train", loss_val, iteration)
            writer.add_scalar("lr", optimizer.get_lr()[0], iteration)
            ious = calculate_iou(prediction, label)
            writer.add_scalars("iou/train", {k: v for k, v in zip(classes, ious)}, iteration)

        if iteration % args.log_interval == 0 and rank == 0:
            now = time()
            etime = (now - start_time) / 60
            print(f"Iteration: {iteration}, Loss: {loss_val}, Elapsed Time: {etime}")
        
        if iteration % args.checkpoint_interval == 0 and rank == 0:
            model = model.eval()
            evaluate_model(model, validation_generator, writer, args.val_iters, iteration)
            model = model.train()
            save_folder = Path("checkpoint_lovasz")
            save_folder.mkdir(exist_ok=True)
            name = f"model_{iteration}.pth"
            savepath = save_folder / name
            print(f"Saving model on iteration {iteration} to {savepath}")
            save_ckpt(cfg, model, optimizer, None, savepath)
        
        if iteration > MAX_ITERS:
            save_folder = Path("checkpoint_lovasz")
            save_folder.mkdir(exist_ok=True)
            save_ckpt(cfg, model, optimizer, None, save_folder / f"model_{iteration}.pth")
            break


@torch.no_grad()
def evaluate_model(model, val_dataloader, writer, val_iters, cur_step):
    rank = dist.get_rank()
    if rank == 0:
        print("Start evaluation")
    criterion = smp.losses.LovaszLoss("multiclass")
    ious = []
    losses = []
    for iter, (image, label) in enumerate(val_dataloader):
        if iter >= val_iters:
            break
        image = image.to(rank)
        label = label.to(rank)
        prediction = model(image)
        prediction = F.upsample(prediction, (args.fine_width, args.fine_height), mode="bilinear") 
        loss = criterion(prediction, label)
        loss = loss.item()
        iou = calculate_iou(prediction, label)
        ious.append(iou)
        losses.append(loss)
    if rank == 0:
        ious = np.mean(ious, axis=0)
        writer.add_scalar("loss/val", np.mean(losses), cur_step)
        classes = ["background", "upper", "lower", "whole"]
        writer.add_scalars("iou/val", {k: v for k, v in zip(classes, ious)}, cur_step)
    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-iters", type=int, default=160_000)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--warmup-iters", type=int, default=1500)
    parser.add_argument("--fine-width", type=int, default=768)
    parser.add_argument("--fine-height", type=int, default=768)
    parser.add_argument("--mean", type=float, default=0.5)
    parser.add_argument("--std", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--config", type=str, default="topformer/config.py")
    parser.add_argument("--init-from", type=str, required=True)
    parser.add_argument("--val-iters", type=int, default=50)
    args = parser.parse_args()
    print("WE ARE WITH DICELOSS")
    setup()
    main(args)
