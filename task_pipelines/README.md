참조할만한 레포
- 안드레아 카파치, nanogpt
https://github.com/WongKinYiu/yolov7/blob/main/train.py
https://github.com/CUBOX-Co-Ltd/cifar10_DDP_FSDP_DS
https://github.com/zzh-tech/ESTRNN/blob/master/train/ddp.py
https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#initialize-ddp-with-torch-distributed-run-torchrun
- https://github.com/Jeff-sjtu/HybrIK
- https://github.com/open-mmlab/mmdetection
- https://github.com/pytorch/examples/blob/main/imagenet
- https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
- https://github.com/WongKinYiu/yolov7
- https://github.com/michuanhaohao/AICITY2021_Track2_DMT/
- https://github.com/IgorSusmelj/pytorch-styleguide




### Attention 관련 처리
- nn.MultiheadAttention incompatibility with DDP https://github.com/pytorch/pytorch/issues/26698
```python
model = DDP(model, device_ids=[local_rank], 
                output_device=local_rank,
                find_unused_parameters=isinstance(layer, nn.MultiheadAttention) for layer in model.modules())

```