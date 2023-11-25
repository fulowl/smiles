from .augmentor import Graph, Augmentor, Compose, RandomChoice
from .identity import Identity

from .instance_mask import InstanceMasking
from .instance_replace import InstanceReplace
from .instance_rand import InstanceRand
from .instance_drop import InstanceDrop

__all__ = [
    'Graph',
    'Augmentor',
    'Compose',
    'RandomChoice',
    'Identity',
    'InstanceMasking',
    'InstanceReplace',
    'InstanceRand',
    'InstanceDrop'
]

classes = __all__
