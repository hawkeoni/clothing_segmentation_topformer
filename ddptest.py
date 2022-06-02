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

c = nn.CrossEntropyLoss()
p = torch.rand(2, 5).to(rank)
t = torch.randint(0, 5, (2, )).to(rank)
loss = c(p, t)
print("I am", dist.get_rank(), loss)
handle = dist.all_reduce(loss, op=dist.ReduceOp.SUM, async_op=True)
handle.wait()
if dist.get_rank() == 0:
    print(loss)


# mean_loss = dist.all_reduce(loss, dist.ReduceOp.SUM).item() / world_size
# model = DDP(DummyModel().to(rank), device_ids=[rank])
# dataset = MyDataset()
# loader = tdata.DataLoader(dataset, batch_size=5, 
# sampler=tdata.DistributedSampler(dataset, rank=dist.get_rank()))

# for i, sample in enumerate(loader):
#     print(i)
#     with torch.no_grad():
#         model(sample)