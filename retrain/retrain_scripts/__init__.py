"""
Módulo de scripts para reentrenamiento del modelo YOLO.

Los imports son lazy para evitar cargar dependencias pesadas
(torch, ultralytics) hasta que realmente se necesiten.
"""

__all__ = [
    'data_preparation',
    'model_training',
    'model_validation',
    'utils'
]


def __getattr__(name):
    """Import lazy: los módulos se cargan solo cuando se acceden."""
    if name in __all__:
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
