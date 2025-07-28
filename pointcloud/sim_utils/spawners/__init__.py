from importlib import import_module, resources
from typing import Dict, Type
import inspect


_REGISTRY: Dict[str, Type] = {}


def register(name):
    def wrapper(cls):
        _REGISTRY[name.lower()] = cls
        return cls
    return wrapper


def create(name, *, catalog, sampler, num_objects, **kwargs):
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown Sampler '{name}'")
    cls = _REGISTRY[key]
    signature = inspect.signature(cls)
    filtered_args = {k: v for k, v in kwargs.items() if k in signature.parameters}
    return cls(catalog, sampler, num_objects, **filtered_args)


for file_name in resources.contents(__name__):
    if file_name.endswith(".py") and file_name not in {"__init__.py"}:
        import_module(f"{__name__}.{file_name[:-3]}")
