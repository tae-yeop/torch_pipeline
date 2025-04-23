"""
분산 처리를 위한 샘플러
"""
import math
import torch
import torch.distributed as dist
from torch.utils.data import Sampler

class CustomDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, is_validation=False):
        """
        validation flag 추가
        """
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Distributed package is not available. Please setup distributed environment properly.")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Distributed package is not available. Please setup distributed environment properly.")
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.is_validation = is_validation
        self.epoch = 0

        if self.is_validation:
            # self.num_samples = int(math.ceil((len(self.dataset) - self.rank) / float(self.num_replicas)))
            # self.total_size = self.num_samples * self.num_replicas
            # 전체 데이터수를 world_size로 나눈다 => 랭크별 담당해야할 데이터 갯수
            self.divided = int(len(self.dataset) // self.num_replicas)
            # 안나눠 떨어지는 나머지 데이터 갯수
            remainder = len(self.dataset) % self.num_replicas
            # 랭크별 담당해야할 샘플수, 마지막이면 나머지도 같이
            self.num_samples = self.divided + remainder if self.rank == (self.num_replicas -1) else self.divided
            self.total_size = len(self.dataset)
        else: # 학습일떄
            self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
            self.total_size = self.num_samples * self.num_replicas


    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.is_validation:
            # add extra samples to make it evenly divisible
            indices += indices[:(self.total_size - len(indices))]
            assert len(indices) == self.total_size

        # subsample
        if self.is_validation:
            if self.rank == (self.num_replicas-1):
                # 마지막 랭크라면 끝까지 모두 담당
                indices = indices[self.rank * self.divided : self.total_size]
            else:
                # 자기 랭크 몫만 담당
                indices = indices[self.rank * self.divided : (self.rank + 1) * self.divided]
        else:  # 학습시
            # 만약 num_replicas=3이면  [0, 1, 2, 0, 1, 2, ....]순으로
            # 각자가 데이터를 찜한다
            indices = indices[self.rank:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class WeightedSampler:
    def __init__(self, datasets, weights, num_samples, replacement=True):
        self.datasets = datasets
        self.weights = weights
        self.num_samples = num_samples
        self.replacement = replacement

    def get_sample_indices(self, dataset_idx):
        # 각 데이터셋의 샘플링 비율 계산
        dataset_weight = self.weights[dataset_idx]
        dataset_samples = int(self.num_samples * dataset_weight)
        
        # 샘플링 인덱스 생성
        if self.replacement:
            return torch.randint(len(self.datasets[dataset_idx]), (dataset_samples,), dtype=torch.int64)
        else:
            return torch.randperm(len(self.datasets[dataset_idx]))[:dataset_samples]

    def __iter__(self):
        total_indices = []
        for i, _ in enumerate(self.datasets):
            indices = self.get_sample_indices(i)
            total_indices.extend(indices)
        
        random.shuffle(total_indices)  # 전체 샘플링 인덱스를 랜덤하게 섞음
        return iter(total_indices)

    def __len__(self):
        return self.num_samples


class DistributedWeightedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, replacement=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.replacement = replacement


    def calculate_weights(self, targets):
        class_sample_count = torch.tensor(
            [(targets == t).sum() for t in torch.unique(targets, sorted=True)])
        weight = 1. / class_sample_count.double()
        samples_weight = torch.tensor([weight[t] for t in targets])
        return samples_weight

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            # [10, 0, 13, 18, 17, 19, 3, 1, ...]
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            # [0,1,2,3,...]
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # get targets (you can alternatively pass them in __init__, if this op is expensive)
        targets = self.dataset.targets
        targets = targets[self.rank:self.total_size:self.num_replicas]
        assert len(targets) == self.num_samples
        weights = self.calculate_weights(targets)

        return iter(torch.multinomial(weights, self.num_samples, self.replacement).tollist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch