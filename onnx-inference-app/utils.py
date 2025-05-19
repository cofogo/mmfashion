import os

def resolve_model_path(base_path: str) -> str:
    """Prepends the local prefix if not running in Docker."""
    if not os.path.exists('/.dockerenv'):
        return os.path.join('onnx-inference-app', base_path)
    return base_path