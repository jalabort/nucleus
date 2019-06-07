import enum


__all__ = [
    'BasketballDetectionsLabelColorMap',
    'BasketballJerseysLabelColorMap'
]


class BasketballDetectionsLabelColorMap(enum.Enum):
    UNKNOWN = 'black'
    PLAYER = 'blue'
    REFEREE = 'green'
    BALL = 'red'
    SUBSTITUTE = 'violet'


class BasketballJerseysLabelColorMap(enum.Enum):
    UNKNOWN = 'black'
    OCCLUDED = 'blue'
    PARTIAL = 'green'
    VISIBLE = 'red'
