
class DataUnit:
    def __init__(self):
        self.HueMin = 0.0
        self.HueMax = 1.0
        self.SaturationMin = 0.0
        self.SaturationMax = 1.0
        self.BrightnessMin = 0.0
        self.BrightnessMax = 1.0
        self.LabelName = ""
        self.strategyname = 8
        # 存儲所有顏色範圍的列表
        self.HSVColorRange = []  
        # 定義標籤名稱列表
        label_names = [
            "orange", "yellow", "blue", "green",
            "black", "red", "white", "others"
        ]
        # 初始化 HSV 顏色範圍
        for label in label_names:
            self.HSVColorRange.append(self.create_color_range(label))

    def create_color_range(self, label_name):
        """建立 HSV 顏色範圍的物件"""
        color_range = {
            "HueMax": 1.0,
            "HueMin": 0.0,
            "SaturationMax": 1.0,
            "SaturationMin": 0.0,
            "BrightnessMax": 1.0,
            "BrightnessMin": 0.0,
            "LabelName": label_name
        }
        return color_range