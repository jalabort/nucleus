from enum import Enum

from nucleus.utils import export 


@export
class BasketballDetectionsLabelColorMap(Enum):
    UNKNOWN = 'black'
    PLAYER = 'blue'
    REFEREE = 'green'
    BALL = 'red'
    SUBSTITUTE = 'violet'


@export
class BasketballJerseysLabelColorMap(Enum):
    UNKNOWN = 'black'
    OCCLUDED = 'blue'
    PARTIAL = 'green'
    VISIBLE = 'red'
