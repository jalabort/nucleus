import enum


class BasketballDetectionLabelColorMap(enum.Enum):
    UNKNOWN = 'black'
    PLAYER = 'blue'
    REFEREE = 'green'
    BALL = 'red'
    SUBSTITUTE = 'violet'


class BasketballJerseyLabelColorMap(enum.Enum):
    UNKNOWN = 'black'
    OCCLUDED = 'blue'
    PARTIAL = 'green'
    VISIBLE = 'red'
