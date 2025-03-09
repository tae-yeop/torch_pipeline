import torch
import torch.nn as nn
from torchmetrics import Metric
from typing import Optional

class MyAccuracy(Metric):
    """Example: A custom Accuracy Metric using torchmetrics."""
    is_differentiable: Optional[bool] = False
    higher_is_better: Optional[bool] = True

    def __init__(self, dist_sync_on_step: bool = False):
        """
        dist_sync_on_step=True 로 설정하면, 
        update()가 호출될 때마다 모든 프로세스가 
        state를 자동으로 합산(혹은 reduce)하여 동기화합니다.
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        # add_state로 정의한 텐서들은 분산 환경에서 
        # 자동으로 합산/평균/연결(cat) 등의 reduce 연산을 수행할 수 있음.
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total",   default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        preds와 target을 받아서 accuracy 통계를 업데이트.
        예: preds가 (N, #class) 로 소프트맥스 출력일 경우 argmax를 취해 사용.
        """
        # (예시) 만약 preds가 로짓(logits) 형태면 argmax를 취함
        if preds.dim() > 1 and preds.size(1) > 1:
            preds = torch.argmax(preds, dim=1)
        # preds, target이 같은 shape라고 가정
        assert preds.shape == target.shape

        # batch단위 correct, total 계산
        correct_ = torch.sum(preds == target)
        total_   = target.numel()

        self.correct += correct_.to(self.correct.device)
        self.total   += torch.tensor(total_, device=self.total.device)

    def compute(self) -> torch.Tensor:
        """
        지금까지 update()로 축적된 correct / total 로 정확도 반환.
        분산 환경에선 reduce_fx='sum'으로 인해 self.correct, self.total가 
        모든 프로세스의 결과가 합쳐진 상태로 compute 호출 시점에 존재.
        """
        return self.correct.float() / self.total


import torch.distributed as dist
class MyAccuracyDDP():
    def __init__(self, device: torch.device, ddp: bool=False):
        
        self.correct = 0
        self.total = 0
        self.device = device
        self.ddp = ddp 


    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, logits, targets):
        preds = logits.argmax(dim=1)
        self.correct += (preds == targets).sum().item()
        self.total += targets.size(0)

    def compute(self):
        # manual 모드에서 전역(global) 정확도를 원한다면, DDP all_reduce
        correct_tensor = torch.tensor([self.correct], device=self.device, dtype=torch.long)
        total_tensor   = torch.tensor([self.total],   device=self.device, dtype=torch.long)
        if self.ddp and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_tensor,   op=dist.ReduceOp.SUM)
        correct_all = correct_tensor.item()
        total_all   = total_tensor.item()
        if total_all == 0:
            return 0.0
        return correct_all / total_all



# train_acc_calc = MyAccuracyDDP(device=args.device, ddp=True)

# for batch_idx, (data, target) in enumerate(train_loader):
#     train_acc_calc.reset()
#     ...
#     for batch_idx, (images, targets) in enumerate(train_loader_iter):
#         ...
#         train_acc_calc.update(outputs, targets)


#     train_acc = train_acc_calc.compute()