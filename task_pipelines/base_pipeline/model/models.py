import torch
import torch.nn as nn

from .registry import register_class_function, register_model

# alias
Linear = nn.Linear
silu = F.silu
dropout_fn = None
# 구버전 dropout 이상한거 처리
def set_torch_version_dependency():
    global dropout_fn
    torch_version = version.parser(torch.__version__)
    # Dropout broke in PyTorch 1.11
    if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
        print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
        dropout_fn = nn.Dropout
    if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
        dropout_fn = nn.Dropout1d
    else:
        dropout_fn = nn.Dropout2d
    
set_torch_version_dependency()


@register_class_function("example_label_class")
class Model(nn.Module):
    def __init__(self, hidden_dim: int = 84):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.dropout = dropout_fn
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)  
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

@register_model
def Model1(**kwargs) -> Model:
    return Model(hidden_dim=64, **kwargs)

@register_model
def Model2(**kwargs) -> Model:
    return Model(hidden_dim=256, **kwargs)


def get_model(model_name: str, **kwargs) -> nn.Module:
    """
    모델명(문자열)과 필요한 파라미터(kwargs)를 입력받아 인스턴스 생성
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"모델 {model_name}이(가) MODEL_REGISTRY에 없습니다.")
    return MODEL_REGISTRY[model_name](**kwargs)
