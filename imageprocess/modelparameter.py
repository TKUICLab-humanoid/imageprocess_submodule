from enum import Enum, IntFlag

# 設定影像相關的常數
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 320
DEFAULT_LABEL_MARK_SIZE = 8
HUE_SCROLLBAR_MAX = 360
SATURATION_SCROLLBAR_MAX = 100
BRIGHTNESS_SCROLLBAR_MAX = 100

# 定義標籤模型
class LabelModel(Enum):
    ORANGE = "orange"
    YELLOW = "yellow"
    BLUE = "blue"
    GREEN = "green"
    BLACK = "black"
    RED = "red"
    WHITE = "white"
    OTHER = "other"

# 定義標籤標記（支援位運算）
class LabelMark(IntFlag):
    ORANGE = 0b00000001  # 1
    YELLOW = 0b00000010  # 2
    BLUE = 0b00000100    # 4
    GREEN = 0b00001000   # 8
    BLACK = 0b00010000   # 16
    RED = 0b00100000     # 32
    WHITE = 0b01000000   # 64
    OTHER = 0b10000000   # 128
