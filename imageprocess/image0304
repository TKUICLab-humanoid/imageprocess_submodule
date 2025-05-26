import cv2
import numpy as np
import rclpy
import rclpy.logging
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from tku_msgs.msg import Location,HSVValue
from tku_msgs.srv import HSVInfo,SaveHSV
from rclpy.node import Node
import time
import configparser
from dataclasses import dataclass, field
import configparser
import colorsys
from imageprocess.dataunit import DataUnit


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        # self.color_deep = 256
        self.Data = DataUnit()
        self.hsv_color_ranges = None
        self.path = ""
        self.hsv_table = None
        # 設定顏色範圍的標籤
        self.labels = ["orange", "yellow", "blue", "green", "black", "red", "white", "others"]
        self.HSVColorRange = {label: self.Data.HSVColorRange for label in self.labels}
        self.raw_HSVColorRange = {label: self.Data.HSVColorRange for label in self.labels}
        # print(self.HSVColorRange)
        # self.build_hsv_table()
        ############################  image  #################################
        self.subscription = self.create_subscription(
            Image,
            'image_raw',
            self.image_callback,
            10
        )
        self.subscription  # prevent unused variable warning
        self.subscription_build = self.create_subscription(
            Image,
            'image_raw',
            self.image_callback_build,
            10
        )
        self.subscription_build  # prevent unused variable warning
        self.processed_image = self.create_publisher(Image, 'processed_image', 10)
        self.build_image = self.create_publisher(Image, 'build_image', 10)
        ######################################################################

        ############################  location  ##############################
        self.location_subscription = self.create_subscription(
            Location,
            '/location',
            self.location_callback,
            10
        )
        self.location_subscription  # prevent unused variable warning
        ######################################################################

        ###########################  Save HSV parameter  #####################
        self.save_hsv = self.create_service(SaveHSV, '/SaveHSV', self.save_hsv_callback)
        ######################################################################

        #########################   color model HSV  #########################
        self.color_model_HSV = self.create_subscription(
            HSVValue,
            '/HSVValue_Topic',
            self.color_model_HSV_callback,
            1000
        )
        self.color_model_HSV  # prevent unused variable warning
        ######################################################################

        #########################  buildcolor  ###############################

        #
        self.hsv_load = self.create_service(HSVInfo, '/LoadHSVInfo',  self.load_hsv_info_callback)
        # self.hsv_build = self.create_service(BuildModel, '/BuildModel', self.build_model_callback)
        #
        self.bridge = CvBridge()
        self.check_image_source = False
        self.resized_image = None
        self.location = ""
        # HSV 查詢表
        # self.table_bmp_to_hsv = self.build_hsv_table()

        # 顏色分類的樣本和標籤
        self.bmp_sample = np.random.randint(0, 9, size=(64 * 64 * 64,), dtype=np.uint8)
        self.color_labels = {
            "BlackLabel":   {"label": 1, "color": [255, 0, 255] },   # 粉色
            "BlueLabel":    {"label": 2, "color": [128, 0, 128] },   # 紫色
            "GreenLabel":   {"label": 3, "color": [0, 0, 128]   },   # 深藍
            "OrangeLabel":  {"label": 4, "color": [128, 0, 0]   },   # 深紅
            "RedLabel":     {"label": 5, "color": [0, 255, 255] },   # 黃色
            "YellowLabel":  {"label": 6, "color": [128, 128, 0] },   # 黃綠
            "WhiteLabel":   {"label": 7, "color": [0, 255, 255] },   # 青綠
            "OthersLabel":  {"label": 8, "color": [255, 0, 128] },   # 紫粉
        }
    #

    ##############################  load hsv  ########################
    def load_hsv_info_callback(self, request, response):
        """強制從 .ini 更新後，再回應 HSV 參數"""
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



    def color_model_HSV_callback(self, msg):
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

        # ✅ 讓影像即時更新 (🔥 只更新 `self.select_color`)
        if self.check_image_source and self.build_imageprocess is not None:
            self.result_image = self.color_modeling(self.build_imageprocess, update_color=self.select_color)

            # 發佈最新影像
            self.build_image.publish(self.bridge.cv2_to_imgmsg(self.result_image, encoding='bgr8'))

    ##############################  save hsv  #################################
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
    ###########################################################################



    def image_callback(self, msg):
        try:
            # 接收並轉換 ROS 影像到 OpenCV 格式
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # 縮放影像
            self.imageprocess = cv2.resize(self.cv_image, (320, 240))
            self.check_image_source = True
            self.processed_image.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")
        return self.imageprocess



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

    def color_modeling(self, image, update_color=None):
        
        if self.hsv_color_ranges is None:
            print("Error: HSV Color Ranges not loaded!")
            return image  # 避免錯誤，直接回傳原圖

        # ✅ 修正 HSV 轉換的歸一化範圍
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32) / [180, 255, 255]
        self.result_image = image.copy()

        # 🔹 定義 8 種標籤對應的顏色 (BGR 格式)
        label_colors = {
            "black": (255, 0, 255),   # 紫色
            "blue": (128, 0, 128),    # 深藍
            "green": (128, 0, 0),     # 深綠
            "orange": (0, 0, 128),    # 深橘
            "red": (0, 255, 255),     # 天藍
            "yellow": (0, 128, 128),  # 黃色
            "white": (255, 255, 0),   # 淺藍
            "others": (128, 0, 255),  # 粉色
        }

        self.build_image.publish(self.bridge.cv2_to_imgmsg(self.result_image, encoding='bgr8'))

        return self.result_image

    def image_callback_build(self, msg):
        """ROS2 訂閱攝影機影像，並進行分類"""
        try:
            # 轉換 ROS 影像為 OpenCV 格式
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # 縮放影像
            self.build_imageprocess = cv2.resize(self.cv_image, (320, 240))
            
            # 確保 HSV 設定已載入後再分類
            if self.hsv_color_ranges:
                self.result_image = self.color_modeling(self.build_imageprocess)

                # 發佈處理後的影像
                self.build_image.publish(self.bridge.cv2_to_imgmsg(self.result_image, encoding='bgr8'))

            self.check_image_source = True
        except Exception as e:
            print(f"Error in image_callback_build: {e}")

    def ChangeToColorModel(self, colormodel):
        # 將影像分類並生成新影像和標籤模型
        image_height, image_width, _ = colormodel.shape
        b, g, r = cv2.split(colormodel)
        index = (r * 64 * 64 + g * 64 + b).astype(np.int32)
        tmp_values = self.bmp_sample[index]

        result_image = np.zeros_like(colormodel)
        label_model = np.zeros((image_height, image_width), dtype=np.uint8)

        for _, data in self.color_labels.items():
            mask = (tmp_values == data["label"])
            result_image[mask] = data["color"]
            label_model[mask] = data["label"]

        mask_others = (tmp_values == 0)
        result_image[mask_others] = [0, 0, 0]
        label_model[mask_others] = 0x00

        return result_image, label_model

##############################################################################################################

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    # image_subscriber.hsv_building_color_model(image_subscriber.color_labels)
    rclpy.spin(image_subscriber)

    # image_subscriber.color_modeling(image_subscriber.cv_image, "red", image_subscriber.HSVColorRange)
    # image_subscriber.location_callback(image_subscriber.location)

    # image_subscriber.loading_colorbuild_file(image_subscriber.location)
    # try:
    #     while rclpy.ok():
    #         # rclpy.spin(image_subscriber)
    #         # if image_subscriber.check_image_source:
    #             # 獲取縮放影像並分類
    #             # colormodel = image_subscriber.resized_image.copy()
    #             # result_image, label_model = image_subscriber.ChangeToColorModel(colormodel)

    #             # 顯示分類後的影像
    #             # cv2.imshow("Processed Image", result_image)

    #             # 發佈分類後的影像
    #             # ros_image_msg = image_subscriber.bridge.cv2_to_imgmsg(image_subscriber.cv_image, encoding='bgr8')
    #             # image_subscriber.publisher.publish(ros_image_msg)
    #         # self.build_image.publish(self.bridge.cv2_to_imgmsg(self.result_image, encoding='bgr8'))

    #         # image_subscriber.build_image.publish(image_subscriber.bridge.cv2_to_imgmsg(image_subscriber.result_image, encoding='bgr8'))
    #         # image_subscriber.color_modeling(image_subscriber.build_imageprocess, image_subscriber.HSVColorRange)
    #         # image_subscriber.check_image_source = False

    # finally:
    #     image_subscriber.destroy_node()
    #     rclpy.shutdown()
    #     # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
