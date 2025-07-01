import numpy as np
from bson.binary import Binary
import zlib
import bson.json_util as ju
import bson

def compress_array(arr: np.ndarray) -> Binary:
    """Devuelve un Binary BSON comprimido con zlib."""
    return Binary(zlib.compress(arr.astype(np.float32).tobytes()))


def make_serializable(obj):
    """NumPy → tipos nativos recursivamente."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj

def to_bin(arr, dtype=np.float32):
    """→ Binary zlib para guardar en BSON."""
    return Binary(zlib.compress(np.asarray(arr, dtype=dtype).tobytes()))

def from_bin(b, dtype):
    return np.frombuffer(zlib.decompress(bson.Binary(b)), dtype=dtype)