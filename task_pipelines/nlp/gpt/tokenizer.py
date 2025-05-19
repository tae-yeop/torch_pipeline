from abc import ABC, abstractmethod

class BaseTokenizer(ABC):
    """
    encode(list[str]) -> np.ndarray[int64]
    decode(np.ndarray) -> list[str]
    """
    @abstractmethod
    def encode(self, texts): ...
    @abstractmethod
    def decode(self, ids): ...

# ------------------------------------------------------------------
# 1) 29‑way char‑level 토크나이저
class CharTokenizer(BaseTokenizer):
    def __init__(self, max_length=149):
        self.max_length = max_length
        letters = [*'abcdefghijklmnopqrstuvwxyz', ' ', 'cls', 'pad']
        self.stoi   = {ch:i for i,ch in enumerate(letters)}
        self.itos   = {i:ch for ch,i in self.stoi.items()}
        self.cls_id = self.stoi['cls']; self.pad_id = self.stoi['pad']

    def encode(self, texts):
        
    def decode(self, ids):