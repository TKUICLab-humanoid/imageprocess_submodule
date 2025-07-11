import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np
from tku_msgs.msg import Location,HSVValue,DrawImage
from tku_msgs.srv import HSVInfo,SaveHSV
from imageprocess.dataunit import DataUnit
import configparser
import os
from dataclasses import dataclass, field

# 定義顏色範圍的資料類別
@dataclass
class ColorRange:
    HueMax: float = 1.0
    HueMin: float = 0.0
    SaturationMax: float = 1.0
    SaturationMin: float = 0.0
    BrightnessMax: float = 1.0
    BrightnessMin: float = 0.0
    LabelName: str = ""

class Imageprocess(Node):
    def __init__(self):
        super().__init__("imageprocess")  # 修正點
        self.labels = ["orange", "yellow", "blue", "green", "black", "red", "white", "others"]
        self.get_logger().info("imageprocess node has been started")
        self.image_subscriber = self.create_subscription(Image, "image_raw", self.origin_to_cv, 10)
        self.image_publisher = self.create_publisher(Image, "build_image", 10)
        self.build_publisher = self.create_publisher(Image, "processed_image", 10)
        self.bridge = CvBridge()
        self.cv_image = None
        self.HSVColorRange = {label: ColorRange(LabelName=label) for label in self.labels}
        self.hsv_lookup_table = None

        # ✅ 定義顏色標籤對應的 BGR 顏色
        self.label_colors = {
            "black": (255, 0, 255),   # 紫色
            "blue": (128, 0, 128),    # 深藍
            "green": (128, 0, 0),     # 深綠
            "orange": (0, 0, 128),    # 深橘
            "red": (0, 255, 255),     # 天藍
            "yellow": (0, 128, 128),  # 黃色
            "white": (255, 255, 0),   # 淺藍
            "others": (128, 0, 255),  # 粉色
        }
        ############################  location  ##############################
        self.location_subscription = self.create_subscription(
            Location,
            '/location',
            self.location_callback,
            10
        )
        self.location_subscription  # prevent unused variable warning
        ############################  location  ##############################

        #########################   color model HSV  #########################
        self.color_model_HSV = self.create_subscription(
            HSVValue,
            '/HSVValue_Topic',
            self.color_model_HSV_callback,
            1000
        )
        self.color_model_HSV  # prevent unused variable warning
        #########################   color model HSV  #########################

        #########################   color model HSV  #########################
        self.hsv_load = self.create_service(HSVInfo, '/LoadHSVInfo',  self.load_hsv_info_callback)
        #########################   color model HSV  #########################

        #########################   color model HSV  #########################
        self.save_hsv = self.create_service(SaveHSV, '/SaveHSV', self.save_hsv_callback)
        #########################   color model HSV  #########################
        self.draw_requests = []
        self.draw_sub = self.create_subscription(
            DrawImage,
            '/draw_image',
            self.draw_image_callback,
            10
        )

    def location_callback(self, msg):
        """讀取 HSV 參數，更新顏色範圍"""
        print(f"Received location: {msg.data}")
        self.path = f"{msg.data}/ColorModelData.ini"
        config = configparser.ConfigParser()
        config.read(self.path)

        key_mapping = {
            "hue_max": "HueMax", "hue_min": "HueMin",
            "saturation_max": "SaturationMax", "saturation_min": "SaturationMin",
            "brightness_max": "BrightnessMax", "brightness_min": "BrightnessMin"
        }

        for label in self.HSVColorRange:
            if label in config:
                section = {
                    key_mapping[k]: float(v) * (360 if 'hue' in k else 100)
                    for k, v in config[label].items() if k in key_mapping and v.replace('.', '', 1).isdigit()
                }
                if isinstance(self.HSVColorRange[label], dict):
                    self.HSVColorRange[label].update(section)
                else:
                    for k, v in section.items():
                        setattr(self.HSVColorRange[label], k, v)

        # print("HSV Color Ranges:", self.HSVColorRange)

        # ✅ 把 ColorRange 轉成 tuple，供 color_modeling() 使用
        self.hsv_color_ranges = {
            label: (
                data.HueMin / 360, data.HueMax / 360,
                data.SaturationMin / 100, data.SaturationMax / 100,
                data.BrightnessMin / 100, data.BrightnessMax / 100
            )
            for label, data in self.HSVColorRange.items()
        }

        print("HSV Color Model Updated and Waiting for Camera Input...")

    def color_model_HSV_callback(self, msg):
        #隨時更新HSV值
        """更新並儲存 HSV 參數 (從 UI 接收)，並即時更新分類"""
        self.get_logger().info(f"Received HSV Value: {msg}")

        # 更新 `self.HSVColorRange` (原始存儲格式)
        self.HSVColorRange[self.select_color].HueMax = msg.hmax
        self.HSVColorRange[self.select_color].HueMin = msg.hmin
        self.HSVColorRange[self.select_color].SaturationMax = msg.smax
        self.HSVColorRange[self.select_color].SaturationMin = msg.smin
        self.HSVColorRange[self.select_color].BrightnessMax = msg.vmax
        self.HSVColorRange[self.select_color].BrightnessMin = msg.vmin

        # ✅ 轉換成 tuple，讓 `color_modeling()` 能正確使用
        self.hsv_color_ranges[self.select_color] = (
            msg.hmin / 360, msg.hmax / 360,
            msg.smin / 100, msg.smax / 100,
            msg.vmin / 100, msg.vmax / 100
        )

        print(f"Updated computed HSV data for {self.select_color}: {self.hsv_color_ranges[self.select_color]}")
        self.get_matching_rgb_from_lut(self.select_color)

    def load_hsv_info_callback(self, request, response):
        #UI選擇顏色後，回傳該顏色的ini檔案對應的HSV值
        print(self.HSVColorRange)  # Debug
        self.select_color = request.colorlabel
        color_data = self.HSVColorRange.get(request.colorlabel)
        print(f"Retrieved color_data: {color_data}")

        if color_data:
            # 這裡 **只在回應時乘 360 或 100**，而不會影響 `self.HSVColorRange` 內的數據
            response.hmin = int(color_data.HueMin)  # **確保只乘一次**
            response.hmax = int(color_data.HueMax)
            response.smin = int(color_data.SaturationMin)
            response.smax = int(color_data.SaturationMax)
            response.vmin = int(color_data.BrightnessMin)
            response.vmax = int(color_data.BrightnessMax)

            print(f"Returning response: hmin={response.hmin}, hmax={response.hmax}, "
                f"smin={response.smin}, smax={response.smax}, vmin={response.vmin}, vmax={response.vmax}")
        else:
            print(f"Warning: Color label '{request.colorlabel}' not found!")

        return response


    def save_hsv_callback(self, request, response):
        """儲存 HSV 參數到 .ini 檔案"""
        if request.save:
            config = configparser.ConfigParser()
            
            for label, data in self.HSVColorRange.items():
                config[label] = {k: str(getattr(data, v) / (360 if 'Hue' in v else 100)) 
                                for k, v in {
                                    "hue_max": "HueMax", "hue_min": "HueMin",
                                    "saturation_max": "SaturationMax", "saturation_min": "SaturationMin",
                                    "brightness_max": "BrightnessMax", "brightness_min": "BrightnessMin"
                                }.items()}
            
            with open(self.path, "w") as configfile:
                config.write(configfile)
            
            print(f"Saved HSV data to: {self.path}")
            response.already = True
        else:
            print("Error: Could not save HSV data to .ini file!")
            response.already = False
        return response


    def origin_to_cv(self, msg):
        try:
            # print("origin_to_cv")
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.process_image()  # 加上這一行，確保每次接收到影像時都執行 process_image
            self.build_image()  # 加上這一行，確保每次接收到影像時都執行 build_image

        except Exception as e:
            print(e)

    def process_image(self):
        if self.cv_image is not None:
            # print("Displaying real-time video")
            self.hsv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
            self.image_publisher.publish(self.bridge.cv2_to_imgmsg(self.hsv_image, "bgr8"))
    
    def build_image(self):
        if self.cv_image is not None:
            # print("Displaying real-time video")
            self.hsv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
            self.build_publisher.publish(self.bridge.cv2_to_imgmsg(self.hsv_image, "bgr8"))
            return self.hsv_image



    def create_hsv_lookup_table(self):
        print("🚀 建立 HSV Lookup Table...")

        # ✅ 建立 RGB 色彩空間的所有組合 (256, 256, 256, 3)
        rgb_values = np.stack(np.meshgrid(np.arange(256), np.arange(256), np.arange(256), indexing="ij"), axis=-1).astype(np.uint8)

        # ✅ 批量轉換為 HSV
        self.hsv_lookup_table = cv2.cvtColor(rgb_values.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(256, 256, 256, 3)

        print("✅ HSV Lookup Table 建立完成！")
        return self.hsv_lookup_table

    def get_matching_rgb_for_all_labels(self):
        """
        透過 HSV 查表 (LUT) 來獲取符合 8 種標籤 HSV 範圍的 RGB 顏色
        """
        matching_rgb_per_label = {}

        for color_label, (h_min, h_max, s_min, s_max, v_min, v_max) in self.hsv_color_ranges.items():
            hsv_values = self.hsv_lookup_table  # 取得 LUT
            h_mask = (hsv_values[..., 0] >= h_min) & (hsv_values[..., 0] <= h_max)
            s_mask = (hsv_values[..., 1] >= s_min * 255) & (hsv_values[..., 1] <= s_max * 255)
            v_mask = (hsv_values[..., 2] >= v_min * 255) & (hsv_values[..., 2] <= v_max * 255)

            matching_rgb = np.argwhere(h_mask & s_mask & v_mask)  # 找出符合條件的 (r, g, b)
            matching_rgb_per_label[color_label] = [tuple(rgb) for rgb in matching_rgb]

        print("✅ 找到符合 8 種顏色標籤的 RGB 值")
        return matching_rgb_per_label
    def process_image_with_labels(self, image):
        """
        根據 8 種標籤 HSV 範圍來分類影像
        """
        height, width, _ = image.shape
        labeled_image = np.zeros((height, width, 3), dtype=np.uint8)

        # ✅ 取得符合 8 種標籤的 RGB 值
        matching_rgb_per_label = self.get_matching_rgb_for_all_labels()

        for y in range(height):
            for x in range(width):
                r, g, b = image[y, x]   # 讀取原始 RGB

                # 🔹 檢查該像素是否符合某個標籤的 RGB 範圍
                for color_label, rgb_list in matching_rgb_per_label.items():
                    if (r, g, b) in rgb_list:
                        labeled_image[y, x] = self.label_colors[color_label]  # ✅ 使用標籤對應的 BGR 顏色
                        break
                else:
                    labeled_image[y, x] = self.label_colors["others"]  # ✅ 預設為 "others"

        return labeled_image



def main(args=None):
    rclpy.init(args=args)
    imageprocess = Imageprocess()
    imageprocess.create_hsv_lookup_table()
    while rclpy.ok():
        processed_image = imageprocess.build_image()
        if processed_image is not None:
            imageprocess.process_image_with_labels(processed_image)
        else:
            print("⚠️ 影像為 None，跳過本次處理...")

    # imageprocess.lookup_hsv_value(255, 255, 255, imageprocess.create_hsv_lookup_table())
    # print
    rclpy.spin(imageprocess)
    rclpy.shutdown()
if __name__ == '__main__':
    main()