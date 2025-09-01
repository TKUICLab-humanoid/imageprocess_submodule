import cv2
import numpy as np
import rclpy
import rclpy.logging
from sensor_msgs.msg import Image,CompressedImage
from cv_bridge import CvBridge
from tku_msgs.msg import HSVValue,DrawImage
from tku_msgs.srv import HSVInfo,SaveHSV
from rclpy.node import Node
import time
import configparser
from dataclasses import dataclass, field
import colorsys
import json

from std_msgs.msg import String,UInt8MultiArray,MultiArrayDimension
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

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
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

    def __init__(self):
        super().__init__('image_subscriber')

        # -----------------------------
        # 影像/HSV 初始狀態（保留原名）
        # -----------------------------
        self.hsv = None
        self.lower = None
        self.upper = None

        self.path = ""
        self.hsv_table = None

        # 設定顏色範圍的標籤（保留原名）
        self.labels = ["orange", "yellow", "blue", "green", "black", "red", "white", "others"]
        self.color_labels = {
            "BlackLabel":   {"label": 1, "color": [255,   0, 255]},   # 粉
            "BlueLabel":    {"label": 2, "color": [128,   0, 128]},   # 紫
            "GreenLabel":   {"label": 3, "color": [  0,   0, 128]},   # 深藍
            "OrangeLabel":  {"label": 4, "color": [128,   0,   0]},   # 深紅
            "RedLabel":     {"label": 5, "color": [255, 255,   0]},   # 黃
            "YellowLabel":  {"label": 6, "color": [128, 128,   0]},   # 黃綠
            "WhiteLabel":   {"label": 7, "color": [  0, 255, 255]},   # 青綠
            "OthersLabel":  {"label": 8, "color": [255,   0, 128]},   # 紫粉
        }

        # 顏色參數表（保留原名）
        self.HSVColorRange = {label: ColorRange(LabelName=label) for label in self.labels}

        # -----------------------------
        # QoS（集中管理）
        # -----------------------------
        # 只保留最新一筆；若要晚加入者可拿到上一筆，把 durability 改成 TRANSIENT_LOCAL
        qos_latest = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )
        qos_img = 10  # 依你原本設定，影像 topic 用 depth=10

        # -----------------------------
        # Publisher / Subscriber（保留原名）
        # -----------------------------
        # Publisher: 全色總表（JSON）
        self.info_pub = self.create_publisher(String, 'object_info', qos_latest)

        # 每個顏色各一個 Publisher：detections/<label>
        self.det_pubs = {
            label: self.create_publisher(String, f'detections/{label}', qos_latest)
            for label in self.labels
        }

        # 影像 I/O
        self.subscription = self.create_subscription(
            Image, '/image_raw', self.image_callback, qos_img
        )
        self.zoom_in = self.create_publisher(Image, 'zoom_in', qos_img)
        self.processed_image = self.create_publisher(Image, 'processed_image', qos_img)
        self.build_image = self.create_publisher(Image, 'build_image', qos_img)
        self.mask_pub = self.create_publisher(Image, 'mask_image', qos_img)

        # color label（保留原名）
        self.label_pub = self.create_publisher(UInt8MultiArray, 'label_matrix', qos_img)

        # color model HSV（保留原名）
        self.color_model_HSV = self.create_subscription(
            HSVValue, '/HSVValue_Topic', self.color_model_HSV_callback, 1000
        )

        # DrawImage（保留原名）
        self.draw_requests = []
        self.draw_sub = self.create_subscription(
            DrawImage, '/drawimage', self.draw_image_callback, qos_img
        )

        # -----------------------------
        # Services（保留原名）
        # -----------------------------
        self.save_hsv = self.create_service(SaveHSV, '/SaveHSV', self.save_hsv_callback)
        self.hsv_load = self.create_service(HSVInfo, '/LoadHSVInfo', self.load_hsv_info_callback)

        # -----------------------------
        # 參數（支援動態更新，保留原名）
        # -----------------------------
        self.declare_parameter('location', 'ar')
        self.declare_parameter('zoom_in', 1.0)
        # 讀取初始值（保留原名）
        loc = self.get_parameter('location').value
        self.zoomin = self.get_parameter('zoom_in').get_parameter_value().double_value

        # 初始化時依照 location 套用
        # 你的 location_callback 目前吃字串就可直接用；若吃 msg 請改成能吃 str 的版本或包一層
        self.location_callback(loc)

        # 允許執行中動態變更 location / zoom_in
        self.add_on_set_parameters_callback(self._on_param_update)

        # -----------------------------
        # 其他狀態（保留原名）
        # -----------------------------
        self.bridge = CvBridge()
        self.check_image_source = False
        self.resized_image = None
        self.location = ""  # 若你的其他程式需要此欄位，先保留

        self.get_logger().info('image_subscriber initialized ✅')


    # -------------------------------------------------
    # 參數動態更新 callback（新增的私有方法）
    # -------------------------------------------------
    def _on_param_update(self, params):
        from rclpy.parameter import Parameter
        updated = False
        for p in params:
            if p.name == 'location' and p.type_ == Parameter.Type.STRING:
                try:
                    self.location_callback(p.value)
                    updated = True
                except Exception as e:
                    self.get_logger().error(f'Failed to apply location "{p.value}": {e}')
            elif p.name == 'zoom_in' and p.type_ in (
                Parameter.Type.DOUBLE, Parameter.Type.INTEGER
            ):
                self.zoomin = float(p.value)
                updated = True

        from rcl_interfaces.msg import SetParametersResult
        return SetParametersResult(successful=True if updated else True)



    def location_callback(self, loc):
        """讀取 location，並初始化 HSVColorRange"""
        print(f"Received location: {loc}")
        self.path = f"/workspace/towen/src/strategy/strategy/{loc}/Parameter/ColorModelData.ini"
        print("path = ", self.path)

        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(self.path)

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
                # 這裡你的 INI 已經是 0~179 / 0~255 的整數域，所以不做歸一化
                updates[attr_name] = raw
            if isinstance(target, dict):
                target.update(updates)
            else:
                for attr_name, val in updates.items():
                    setattr(target, attr_name, val)

    ##############################  load hsv  ########################
    def load_hsv_info_callback(self, request, response):
        print("Loading HSV data from INI file...")
        self.select_color = request.colorlabel
        color_data = self.HSVColorRange.get(request.colorlabel)
        print(f"Retrieved color_data: {color_data}")

        if color_data:
            response.hmin = int(color_data.HueMin)
            response.hmax = int(color_data.HueMax)
            response.smin = int(color_data.SaturationMin)
            response.smax = int(color_data.SaturationMax)
            response.vmin = int(color_data.BrightnessMin)
            response.vmax = int(color_data.BrightnessMax)
            self.lower = np.array([response.hmin, response.smin, response.vmin], dtype=np.uint8)
            self.upper = np.array([response.hmax, response.smax, response.vmax], dtype=np.uint8)
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


    # def image_callback(self, msg: Image):
    def image_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            height, width = cv_img.shape[:2]

            # 中央放大
            new_w = int(width / self.zoomin)
            new_h = int(height / self.zoomin)
            x1 = (width - new_w) // 2
            y1 = (height - new_h) // 2
            x2 = x1 + new_w
            y2 = y1 + new_h

            cropped = cv_img[y1:y2, x1:x2]
            zoomed_frame = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
            self.zoom_in.publish(self.bridge.cv2_to_imgmsg(zoomed_frame, encoding='bgr8'))

            hsv = cv2.cvtColor(zoomed_frame, cv2.COLOR_BGR2HSV)

            if self.lower is None or self.upper is None:
                return

            # 原本單色 build（可留）
            self.build_hsv_table(hsv, zoomed_frame)

            # 取用影像 header 的時間戳，丟給每色 topic
            stamp = {'sec': msg.header.stamp.sec, 'nanosec': msg.header.stamp.nanosec}
            self.build_all_hsv_table(hsv, zoomed_frame, stamp)

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
                hsv, (0, int(s_low), int(v_low)),
                (int(h_high), int(s_high), int(v_high))
            )
            mask2 = cv2.inRange(
                hsv, (int(h_low), int(s_low), int(v_low)),
                (179, int(s_high), int(v_high))
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
        return vis_msg, mask_msg

    def build_all_hsv_table(self, hsv, resized, stamp):
        h, w = hsv.shape[:2]
        total_mask = np.zeros((h, w), dtype=np.uint8)       # 純黑白二值化
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)    # BGR 彩色輸出

        # 總表（我改成精簡欄位，避免太肥）
        detections_all = { label: [] for label in self.HSVColorRange.keys() }

        # 針對每個顏色做 inRange、形態學處理並累積到 total_mask
        for label, color_obj in self.HSVColorRange.items():
            h_low, h_high = int(color_obj.HueMin), int(color_obj.HueMax)
            s_low, s_high = int(color_obj.SaturationMin), int(color_obj.SaturationMax)
            v_low, v_high = int(color_obj.BrightnessMin), int(color_obj.BrightnessMax)
            if (h_low, h_high, s_low, s_high, v_low, v_high) == (0,0,0,0,0,0):
                # 空設定就跳過
                continue

            # 切分跨過 0 度的情況
            if h_low <= h_high:
                mask_i = cv2.inRange(hsv, (h_low, s_low, v_low), (h_high, s_high, v_high))
            else:
                m1 = cv2.inRange(hsv, (0, s_low, v_low), (h_high, s_high, v_high))
                m2 = cv2.inRange(hsv, (h_low, s_low, v_low), (179, s_high, v_high))
                mask_i = cv2.bitwise_or(m1, m2)

            # 開運算去雜訊
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_i = cv2.morphologyEx(mask_i, cv2.MORPH_OPEN, kernel, iterations=2)
            total_mask = cv2.bitwise_or(total_mask, mask_i)

            # 對應偽彩色
            label_key = label.capitalize() + "Label"
            bgr_color = np.array(self.color_labels.get(label_key, {"color":[255,255,255]})["color"], dtype=np.uint8)
            color_mask[mask_i > 0] = bgr_color

            # 找輪廓並記錄偵測結果
            contours, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 這一色的精簡清單（高頻 topic 用）
            color_list_compact = []

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 50:
                    continue
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
                else:
                    cx, cy = x + w_box//2, y + h_box//2

                item = {
                    "bbox": (x, y, w_box, h_box),
                    "centroid": (cx, cy),
                    "area": float(area),
                    "aspect_ratio": w_box / h_box if h_box>0 else 0,
                    "label": label
                }
                color_list_compact.append(item)
                detections_all[label].append(item)

            # 🆕 每個顏色各自 publish 一則：/detections/<label>
            msg_color = {
                "stamp": stamp,
                "width": w, "height": h,
                "label": label,
                "objects": color_list_compact
            }
            try:
                self.det_pubs[label].publish(String(data=json.dumps(msg_color)))
            except Exception as e:
                self.get_logger().warning(f"publish detections/{label} failed: {e}")

        # ---- 發佈總表 object_info（精簡版）----
        detections_all["_stamp"] = stamp
        self.info_pub.publish(String(data=json.dumps(detections_all)))

        # ---- publish total_mask 二維矩陣 ----
        mat_msg = UInt8MultiArray()
        mat_msg.layout.dim.append(MultiArrayDimension(label='rows', size=h, stride=h*w))
        mat_msg.layout.dim.append(MultiArrayDimension(label='cols', size=w, stride=w))
        mat_msg.data = total_mask.flatten().tolist()
        self.label_pub.publish(mat_msg)

        # ---- 原有的影像可視化發佈 ----
        vis_all = cv2.bitwise_and(color_mask, color_mask, mask=total_mask)

        # 畫圖請求
        if self.draw_requests:
            for req in self.draw_requests:
                color = (int(req['b']), int(req['g']), int(req['r']))
                pt1 = (int(req['xmin']), int(req['ymin']))
                pt2 = (int(req['xmax']), int(req['ymax']))
                if req['mode'] == 0:
                    cv2.line(vis_all, pt1, pt2, color, thickness=2)
                elif req['mode'] == 1:
                    cv2.rectangle(vis_all, pt1, pt2, color, thickness=2)
            self.draw_requests.clear()

        vis_msg_all = self.bridge.cv2_to_imgmsg(vis_all, encoding='bgr8')
        self.build_image.publish(vis_msg_all)
        
    def draw_image_callback(self, msg):
        self.draw_requests.append({
            'cnt': msg.cnt,
            'mode': msg.mode,
            'xmin': msg.xmin,
            'xmax': msg.xmax,
            'ymin': msg.ymin,
            'ymax': msg.ymax,
            'r': msg.rvalue,
            'g': msg.gvalue,
            'b': msg.gvalue
        })
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
        rclpy.spin(image_subscriber)        # ← 單次呼叫，會一直執行 callback
    finally:
        image_subscriber.destroy_node()
        rclpy.shutdown()