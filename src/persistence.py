import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


def json_ready(value):
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def state_value(value, default):
    if isinstance(default, np.ndarray):
        result = np.asarray(value, dtype=default.dtype)
        if not np.isfinite(result).all():
            raise ValueError('State arrays must be finite')
        return result
    if isinstance(default, (int, float)) and not isinstance(default, bool):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError('State numbers must be finite')
        return value
    if default is not None and not isinstance(value, type(default)):
        raise ValueError(f'Expected {type(default).__name__}, got {type(value).__name__}')
    return value


def write_json(path, data, *, indent=2, default=None):
    path = Path(path)
    payload = json.dumps(json_ready(data), indent=indent, allow_nan=False, default=default)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        temporary.write_text(payload + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
