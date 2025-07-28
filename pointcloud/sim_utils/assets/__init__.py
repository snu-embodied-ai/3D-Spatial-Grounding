from importlib import import_module, resources
from typing import Dict, List


_REGISTRY: Dict[str, List] = {}


def register(name):
    def wrapper(asset_list):
        _REGISTRY[name.lower()] = asset_list
        return asset_list
    return wrapper


def create(name):
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown Asset List '{name}'")
    asset_list = _REGISTRY[key]
    return asset_list


for file_name in resources.contents(__name__):
    if file_name.endswith(".py") and file_name not in {"__init__.py"}:
        import_module(f"{__name__}.{file_name[:-3]}")
