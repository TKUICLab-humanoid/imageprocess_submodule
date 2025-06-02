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
import json

from std_msgs.msg import String

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

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.hsv = None
        self.lower = None
        self.upper = None

        # self.color_deep = 256
        self.path = ""
        self.hsv_table = None
        # 設定顏色範圍的標籤
        self.labels = ["orange", "yellow", "blue", "green", "black", "red", "white", "others"]
        self.color_labels = {
            "BlackLabel":   {"label": 1, "color": [255,   0, 255] },   # 粉
            "BlueLabel":    {"label": 2, "color": [128,   0, 128] },   # 紫
            "GreenLabel":   {"label": 3, "color": [  0,   0, 128] },   # 深藍
            "OrangeLabel":  {"label": 4, "color": [128,   0,   0] },   # 深紅
            "RedLabel":     {"label": 5, "color": [255, 255,   0] },   # 黃
            "YellowLabel":  {"label": 6, "color": [128, 128,   0] },   # 黃綠
            "WhiteLabel":   {"label": 7, "color": [  0, 255, 255] },   # 青綠
            "OthersLabel":  {"label": 8, "color": [255,   0, 128] },   # 紫粉
        }
        # Publisher: 物件資訊 (JSON)
        self.info_pub = self.create_publisher(String, 'object_info', 10)
        # ↓↓↓ 這裡重建一個跟 HSVColorRange key 一樣的小寫 map ↓↓↓

        self.HSVColorRange = {label: ColorRange(LabelName=label) for label in self.labels}
        ############################  image  #################################
        self.subscription = self.create_subscription(
            Image,
            'image_raw',
            self.image_callback,
            10
        )
        self.subscription  # prevent unused variable warning
        self.processed_image = self.create_publisher(Image, 'processed_image', 10)
        self.build_image = self.create_publisher(Image, 'build_image', 10)
        self.mask_pub = self.create_publisher(Image, 'mask_image', 10)
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

    def location_callback(self, msg):
        """讀取 HSV 參數，更新顏色範圍"""
        print(f"Received location: {msg.data}")
        self.path = f"{msg.data}/ColorModelData.ini"

        config = configparser.ConfigParser()
        config.optionxform = str                # 保留原始大小寫／底線
        config.read(self.path)

        # INI 裡的 key → 物件屬性名稱
        key_mapping = {
            "hue_max":        "HueMax",
            "hue_min":        "HueMin",
            "saturation_max": "SaturationMax",
            "saturation_min": "SaturationMin",
            "brightness_max": "BrightnessMax",
            "brightness_min": "BrightnessMin"
        }

        for label, target in self.HSVColorRange.items():
            if not config.has_section(label):
                continue

            updates = {}
            for ini_key, attr_name in key_mapping.items():
                if not config.has_option(label, ini_key):
                    continue
                try:
                    raw = config.getfloat(label, ini_key)
                except ValueError:
                    print(f"[WARN] {label}:{ini_key} 不是數字，跳過")
                    continue
                # 如果 raw > 1，代表它還沒被歸一化，才做除法
                if ini_key.startswith("hue_"):
                    norm = raw 
                else:
                    norm = raw 
                updates[attr_name] = norm

            # 寫回 HSVColorRange
            if isinstance(target, dict):
                target.update(updates)
            else:
                for attr_name, val in updates.items():
                    setattr(target, attr_name, val)
    ##############################  load hsv  ########################
    def load_hsv_info_callback(self, request, response):
        print("Loading HSV data from INI file...")
        # """強制從 .ini 更新後，再回應 HSV 參數"""
        # print(self.HSVColorRange)  # Debug
        self.select_color = request.colorlabel
        color_data = self.HSVColorRange.get(request.colorlabel)
        # print(f"Retrieved color_data: {color_data}")

        if color_data:
            response.hmin = int(color_data.HueMin) 
            response.hmax = int(color_data.HueMax)
            response.smin = int(color_data.SaturationMin)
            response.smax = int(color_data.SaturationMax)
            response.vmin = int(color_data.BrightnessMin)
            response.vmax = int(color_data.BrightnessMax)
            self.lower = np.array([response.hmin, response.smin, response.vmin], dtype=np.uint8)
            self.upper = np.array([response.hmax, response.smax, response.vmax], dtype=np.uint8)
            # print(f"Returning response: hmin={response.hmin}, hmax={response.hmax}, "
            #     f"smin={response.smin}, smax={response.smax}, vmin={response.vmin}, vmax={response.vmax}")
        else:
            print(f"Warning: Color label '{request.colorlabel}' not found!")
        return response

    def color_model_HSV_callback(self, msg):
        self.HSVColorRange[self.select_color].HueMax = msg.hmax
        self.HSVColorRange[self.select_color].HueMin = msg.hmin
        self.HSVColorRange[self.select_color].SaturationMax = msg.smax
        self.HSVColorRange[self.select_color].SaturationMin = msg.smin
        self.HSVColorRange[self.select_color].BrightnessMax = msg.vmax
        self.HSVColorRange[self.select_color].BrightnessMin = msg.vmin
        self.lower = np.array([msg.hmin, msg.smin, msg.vmin], dtype=np.uint8)
        self.upper = np.array([msg.hmax, msg.smax, msg.vmax], dtype=np.uint8)

    def image_callback(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            resized = cv2.resize(cv_img, (320, 240))
            hsv     = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            if self.lower is None or self.upper is None:
                return
            else:
                self.build_hsv_table(hsv, resized)
                self.build_all_hsv_table(hsv, resized)
                # self.processed_image.publish(mask_msg)
                # self.build_image.publish(mask_all)
        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

    def build_hsv_table(self, hsv, resized):
            h_low, s_low, v_low = self.lower
            h_high, s_high, v_high = self.upper
            if h_low <= h_high:
                mask = cv2.inRange(
                    hsv,
                    (int(h_low), int(s_low), int(v_low)),
                    (int(h_high), int(s_high), int(v_high))
                )
            else:
                mask1 = cv2.inRange(
                    hsv,
                    (0,        int(s_low), int(v_low)),
                    (int(h_high), int(s_high), int(v_high))
                )
                mask2 = cv2.inRange(
                    hsv,
                    (int(h_low), int(s_low), int(v_low)),
                    (179,      int(s_high), int(v_high))
                )
                mask = cv2.bitwise_or(mask1, mask2)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
            mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')

            key = f"{self.select_color.capitalize()}Label"
            if key in self.color_labels:
                b, g, r = self.color_labels[key]['color']
                color_img = np.zeros_like(resized)
                color_img[:] = (b, g, r)
                colored_mask = cv2.bitwise_and(color_img, color_img, mask=mask)
                color_msg = self.bridge.cv2_to_imgmsg(colored_mask, encoding='bgr8')
                self.processed_image.publish(color_msg)

            vis_msg = self.bridge.cv2_to_imgmsg(resized, encoding='bgr8')
            return vis_msg,mask_msg
    
    def build_all_hsv_table(self, hsv, resized):
        h, w = hsv.shape[:2]
        total_mask = np.zeros((h, w), dtype=np.uint8)       # 純黑白二值化
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)    # BGR 彩色輸出

        # 準備一個 dict 來存每個顏色下所有「偵測到的物件」資訊
        detections = { label: [] for label in self.HSVColorRange.keys() }

        for label, color_obj in self.HSVColorRange.items():
            h_low  = int(color_obj.HueMin)
            h_high = int(color_obj.HueMax)
            s_low  = int(color_obj.SaturationMin)
            s_high = int(color_obj.SaturationMax)
            v_low  = int(color_obj.BrightnessMin)
            v_high = int(color_obj.BrightnessMax)

            if (h_low == 0 and h_high == 0
                and s_low == 0 and s_high == 0
                and v_low == 0 and v_high == 0):
                continue

            if h_low <= h_high:
                mask_i = cv2.inRange(
                    hsv,
                    (h_low,  s_low,  v_low),
                    (h_high, s_high, v_high)
                )
            else:
                mask1 = cv2.inRange(
                    hsv,
                    (0,     s_low,  v_low),
                    (h_high,s_high, v_high)
                )
                mask2 = cv2.inRange(
                    hsv,
                    (h_low, s_low,  v_low),
                    (179,   s_high, v_high)
                )
                mask_i = cv2.bitwise_or(mask1, mask2)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_i = cv2.morphologyEx(mask_i, cv2.MORPH_OPEN, kernel, iterations=2)
            total_mask = cv2.bitwise_or(total_mask, mask_i)

            label_key = label.capitalize() + "Label"
            if label_key in self.color_labels:
                bgr_color = np.array(self.color_labels[label_key]["color"], dtype=np.uint8)
            else:
                bgr_color = np.array([255,255,255], dtype=np.uint8)

            color_mask[mask_i > 0] = bgr_color

            contours, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 50:
                    continue

                x, y, w_box, h_box = cv2.boundingRect(cnt)
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x + w_box // 2, y + h_box // 2

                detections[label].append({
                    "bbox":     (x, y, w_box, h_box),
                    "area":     float(area),
                    "centroid": (cx, cy),
                    "color":    bgr_color.tolist(),
                    "label":    label,
                    "mask":     mask_i[y:y+h_box, x:x+w_box].tolist(),  # 只保留當前物件的 mask
                    "contour":  cnt.tolist(),  # 輪廓點
                    "aspect_ratio": w_box / h_box if h_box > 0 else 0,  # 長寬比例
                })

        info_msg = String()
        info_msg.data = json.dumps(detections)  # 將 dict 序列化成 JSON 字串
        self.info_pub.publish(info_msg)

        vis_all = cv2.bitwise_and(color_mask, color_mask, mask=total_mask)

        # 7) 轉成 ROS Image Msg
        mask_all    = self.bridge.cv2_to_imgmsg(total_mask, encoding='mono8')
        vis_msg_all = self.bridge.cv2_to_imgmsg(vis_all,    encoding='bgr8')
        self.build_image.publish(vis_msg_all)


    ##############################  save hsv  #################################
    def save_hsv_callback(self, request, response):
        """儲存 HSV 參數到 .ini 檔案"""
        if request.save:
            config = configparser.ConfigParser()
            
            for label, data in self.HSVColorRange.items():
                config[label] = {k: str(getattr(data, v)) 
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



##############################################################################################################

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    try:
        while rclpy.ok():
            rclpy.spin(image_subscriber)
            # if image_subscriber.check_image_source:
                # 獲取縮放影像並分類
                # colormodel = image_subscriber.resized_image.copy()
                # result_image, label_model = image_subscriber.ChangeToColorModel(colormodel)
                # 顯示分類後的影像
                # cv2.imshow("Processed Image", result_image)
                # 發佈分類後的影像
                # ros_image_msg = image_subscriber.bridge.cv2_to_imgmsg(image_subscriber.cv_image, encoding='bgr8')
                # image_subscriber.publisher.publish(ros_image_msg)
            # self.build_image.publish(self.bridge.cv2_to_imgmsg(self.result_image, encoding='bgr8'))
            # image_subscriber.build_image.publish(image_subscriber.bridge.cv2_to_imgmsg(image_subscriber.result_image, encoding='bgr8'))
            # image_subscriber.color_modeling(image_subscriber.build_imageprocess, image_subscriber.HSVColorRange)
            # image_subscriber.check_image_source = False
    finally:
        image_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
