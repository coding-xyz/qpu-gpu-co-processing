import math
import itertools
from dataclasses import dataclass
from typing import Dict, Tuple, List

import torch


# ----------------------------
# Utilities: Pauli basis rotations
# ----------------------------
def U_for_basis(b: str, device, dtype=torch.complex64):
    """
    Unitary U s.t. measuring in basis b corresponds to applying U then measuring Z.
    b in {'X','Y','Z'}.
    """
    if b == "Z":
        U = torch.eye(2, device=device, dtype=dtype)
    elif b == "X":
        # H
        U = (1 / math.sqrt(2)) * torch.tensor([[1, 1], [1, -1]], device=device, dtype=dtype)
    elif b == "Y":
        # S^\dagger H
        H = (1 / math.sqrt(2)) * torch.tensor([[1, 1], [1, -1]], device=device, dtype=dtype)
        Sdg = torch.tensor([[1, 0], [0, -1j]], device=device, dtype=dtype)
        U = Sdg @ H
    else:
        raise ValueError(f"Unknown basis {b}")
    return U


def kron_n(mats: List[torch.Tensor]) -> torch.Tensor:
    out = mats[0]
    for m in mats[1:]:
        out = torch.kron(out, m)
    return out


def projector(bit: int, device, dtype=torch.complex64):
    # |0><0| or |1><1|
    if bit == 0:
        return torch.tensor([[1, 0], [0, 0]], device=device, dtype=dtype)
    else:
        return torch.tensor([[0, 0], [0, 1]], device=device, dtype=dtype)


def bitstring_to_bits(s: str) -> List[int]:
    return [0 if c == "0" else 1 for c in s]