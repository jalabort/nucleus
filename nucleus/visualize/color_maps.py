from enum import Enum
from public import public


@public
class BasketballDetectionsLabelColorMap(Enum):
    UNKNOWN = 'black'
    PLAYER = 'blue'
    REFEREE = 'green'
    BALL = 'red'
    SUBSTITUTE = 'violet'


@public
class BasketballJerseysLabelColorMap(Enum):
    UNKNOWN = 'black'
    OCCLUDED = 'blue'
    PARTIAL = 'green'
    VISIBLE = 'red'
