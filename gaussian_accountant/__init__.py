from .accountant import IAccountant
from .gdp import GaussianAccountant
from .rdp import RDPAccountant
from .dp_utils import clip_gradients, add_gaussian_noise, per_sample_gradient_norm

__all__ = [
    "IAccountant",
    "GaussianAccountant",
    "RDPAccountant",
    "clip_gradients",
    "add_gaussian_noise",
    "per_sample_gradient_norm",
    "create_accountant",
]


def create_accountant(mechanism: str) -> IAccountant:
    if mechanism == "gdp":
        return GaussianAccountant()
    elif mechanism == "rdp":
        return RDPAccountant()

    raise ValueError(f"Unexpected accounting mechanism: {mechanism}")
