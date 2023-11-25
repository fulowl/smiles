from .jsd import JSD, DebiasedJSD, HardnessJSD
from .vicreg import VICReg
from .triplet import TripletMargin, TripletMarginSP
from .bagloss import BLoss
from .bloss2 import BarlowTwins
from .losses import Loss

__all__ = [
    'Loss',
    'JSD',
    'DebiasedJSD',
    'HardnessJSD',
    'TripletMargin',
    'TripletMarginSP',
    'VICReg',
    'BarlowTwins'
]

classes = __all__
