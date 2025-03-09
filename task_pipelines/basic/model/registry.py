from typing import Dict, Type, Callable

MODEL_REGISTRY: Dict[str, Callable] = {}
CLASS_REGISTRY: Dict[str, Type] = {}


def register_label_class(name: str):
    """
    클래스를 레지스트리에 등록하는 데코레이터
    이건 key를 등록할 수 있음
    """
    def decorator(cls: Type) -> Type:
        CLASS_REGISTRY[name] = cls
        return cls
    return decorator

def register_model(func: Callable) -> Callable:
    MODEL_REGISTRY[func.__name__] = func
    return func