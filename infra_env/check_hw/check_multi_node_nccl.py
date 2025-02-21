# 통신 되는지 확인

import os
import torch
import torch.distributed as dist

local_rank = int(os.environ['LOCAL_RANK'])
rank = int(os.environ['RANK'])
dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)
device = torch.device('cuda', local_rank)

print(rank, local_rank)