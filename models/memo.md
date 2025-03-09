
### find-unused-parameters가 나올시
'''
RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. This error indicates that your module has parameters that were not used in producing loss.
'''

정말 DDP forward를 돌렸을 때 참여하지 않는 레이어가 존재해서 생긴다.
다음과 같이 하면 어떤 모델이 참여하지 않는지 나온다.

require_grad는 문제의 원인이 아니다. 
eval()도 보통은 dropout 정도니깐 문제의 원인이 아니다. 

```
export TORCH_DISTRIBUTED_DEBUG=DETAIL
torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py
```

### Model안에 Loss(nn.Module)를 넣는 경우
- Percptual loss는 eval을 걸어야 하는데 이게 모델안에 들어가면 loop를 돌때 train(), eval() 이걸 작동시켜서 VGG 모델 행동이 바뀜
- 왠만해선 nn.Module을 상속한 loss는 모델의 forward 안에서 실행하지 않는게 좋겠다.