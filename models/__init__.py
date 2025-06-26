import importlib
import pkgutil
from pathlib import Path

from mmengine.registry import Registry

DeepZMODELS = Registry('DeepZLab')

#
_pkg_dir = Path(__file__).resolve().parent
_prefix  = __name__ + '.'
for m in pkgutil.walk_packages([str(_pkg_dir)], _prefix):
    importlib.import_module(m.name)


def list_models():
    print(DeepZMODELS)
    return list(DeepZMODELS.module_dict)


def build_model(model_name: str,
                **kwargs):
    """
    model_name: 在 Registry 里注册的 key
    kwargs    : 直接透传给模型 __init__
    """
    cfg = dict(type=model_name, **kwargs)
    return DeepZMODELS.build(cfg)