import os
import torch
import torch.nn as nn
import torch.utils.data as tdata
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

world_size = int(os.environ["WORLD_SIZE"])
rank = int(os.environ["LOCAL_RANK"])


class MyDataset(tdata.Dataset):

    def __getitem__(self, index):
        return torch.rand(1, 20)
    
    def __len__(self):
        return 600

class DummyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(20, 30)

    def forward(self, x):
        print("We are", dist.get_rank(), x.shape)
        return self.lin(x)

dist.init_process_group(
    backend="nccl", 
    world_size=world_size, 
    rank=rank)

model = DDP(DummyModel().to(rank), device_ids=[rank])
dataset = MyDataset()
loader = tdata.DataLoader(dataset, batch_size=5, 
sampler=tdata.DistributedSampler(dataset, rank=dist.get_rank()))

for i, sample in enumerate(loader):
    print(i)
    with torch.no_grad():
        model(sample)