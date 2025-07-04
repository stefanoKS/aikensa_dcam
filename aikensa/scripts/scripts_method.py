import numpy as np
import yaml


def scale_translation(T: np.ndarray, scale: float) -> np.ndarray:
    """
    Scale only the translation components (T[0,2], T[1,2]) of any 3×3 matrix.
    """
    if T.shape != (3, 3):
        raise ValueError(f"Expected 3×3 matrix, got {T.shape}")
    Ts = T.copy()
    Ts[0, 2] *= scale
    Ts[1, 2] *= scale
    return Ts

def list_to_16bit_int(bit_list):
    if len(bit_list) > 16:
        raise ValueError("Input list must be 16 bits or fewer")
    
    # Pad to 16 bits by adding leading zeros
    full_bits = [0] * (16 - len(bit_list)) + bit_list

    # Convert to binary string and then to int
    bit_str = ''.join(str(b) for b in full_bits)
    return int(bit_str, 2)

def load_register_map(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    # Invert the mapping: {label: address}
    return {v: int(k) for k, v in data.items()}

def invert_16bit_int(value: int) -> int:
    """
    Inverts all bits of a 16-bit integer.
    Example: 0b0000000000001010 -> 0b1111111111110101
    """
    if not 0 <= value <= 0xFFFF:
        raise ValueError("Input must be a 16-bit unsigned integer (0 to 65535)")

    # XOR with 0xFFFF to flip all 16 bits
    return value ^ 0xFFFF

def random_list(length: int) -> list[int]:
    """
    Generates a random list of 0s and 1s of specified length.
    """
    import random
    if length < 0:
        raise ValueError("Length must be a non-negative integer")
    return [random.randint(0, 1) for _ in range(length)]