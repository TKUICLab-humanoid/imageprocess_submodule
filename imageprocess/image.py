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

from std_msgs.msg import String,UInt8MultiArray,MultiArrayDimension,Int16
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

        # 顏色參數表
        self.HSVColorRange = {label: ColorRange(LabelName=label) for label in self.labels}

        # -----------------------------
        # QoS
        # -----------------------------
        # 只保留最新一筆；若要晚加入者可拿到上一筆，把 durability 改成 TRANSIENT_LOCAL
        qos_latest = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )
        qos_img = 10  # 依你原本設定，影像 topic 用 depth=10

        # -----------------------------
        # Publisher / Subscriber
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

        # color label（保留原名）— 改用 Image (mono8) 發 total_mask
        # self.label_pub = self.create_publisher(UInt8MultiArray, 'label_matrix', qos_img)
        self.label_pub = self.create_publisher(Image, 'label_matrix', 10)

        # color model HSV（保留原名）
        self.color_model_HSV = self.create_subscription(
            HSVValue, '/HSVValue_Topic', self.color_model_HSV_callback, 1000
        )

        # DrawImage（保留原名）
        self.draw_requests = []
        self.draw_sub = self.create_subscription(
            DrawImage, '/drawimage', self.draw_image_callback, qos_img
        )
        self.build_status_sub = self.create_subscription(
            Int16, '/BuildStatus', self.build_status_callback, 10
        )
        self.build_status = 0
        self.select_color = "red"
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

        # ===== 形態學 kernel（一次建好）=====
        self.kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # ===== JSON 類 Topic 的「節流 + 去抖」狀態 =====
        # 最快 20Hz；想更省可以把 0.05 改大（例如 0.1 = 10Hz）
        self._pub_period = 0.05  # 秒
        # 每個顏色一個時間戳 + 總表一個
        self._last_pub_t = {"info": 0.0, **{f"det_{l}": 0.0 for l in self.labels}}
        # 每個顏色上一份 payload（字串）—用來判斷「內容是否有變化」
        self._last_payload = {l: "" for l in self.labels}
        # 總表上一份 payload（字串）
        self._last_info_payload = ""

        # （可選）影像類 Topic 也做節流：build_image / mask_image / zoom_in
        self._img_period = 0.001  # 秒，100Hz
        self._last_img_pub = {"build_image": 0.0, "mask_image": 0.0, "zoom_in": 0.0}

        self.get_logger().info('image_subscriber initialized ✅')


    def build_status_callback(self, msg):
        print("BuildStatus:", msg.data)
        self.build_status = msg.data
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


    def image_callback(self, msg):
        try:
            # 1) ROS Image → OpenCV
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            height, width = cv_img.shape[:2]

            # 2) 中央放大（保持你原本 zoom 行為）
            new_w = int(width / self.zoomin)
            new_h = int(height / self.zoomin)
            x1 = (width - new_w) // 2
            y1 = (height - new_h) // 2
            x2 = x1 + new_w
            y2 = y1 + new_h

            cropped = cv_img[y1:y2, x1:x2]
            zoomed_frame = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)

            # zoom_in 影像做節流（避免每幀都發）
            now = time.time()
            if now - self._last_img_pub["zoom_in"] >= self._img_period:
                zoom_msg = self.bridge.cv2_to_imgmsg(zoomed_frame, encoding='bgr8')
                try:
                    zoom_msg.header.stamp.sec = int(msg.header.stamp.sec)
                    zoom_msg.header.stamp.nanosec = int(msg.header.stamp.nanosec)
                except Exception:
                    pass
                self.zoom_in.publish(zoom_msg)
                self._last_img_pub["zoom_in"] = now

            # 3) 低解析度處理（大幅省 CPU）：在小圖上做 HSV/inRange/CCWS
            proc_w, proc_h = 320, 240
            proc_frame = cv2.resize(zoomed_frame, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
            hsv_proc = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2HSV)

            if self.build_status == 1:
                # 4) 單色 build（僅在 lower/upper 已設定時跑；保持你原本邏輯）
                if self.lower is not None and self.upper is not None:
                    self.build_hsv_table(hsv_proc, proc_frame)
            elif self.build_status == 0:
                # 5) 取 header 時間戳，丟給多色偵測（節流＋去抖在函式內處理）
                stamp = {'sec': msg.header.stamp.sec, 'nanosec': msg.header.stamp.nanosec}
                self.build_all_hsv_table(hsv_proc, proc_frame, stamp)

        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")



    def build_hsv_table(self, hsv, resized):
        H_MAX, S_MAX, V_MAX = 179, 255, 255

        h_low, s_low, v_low = map(int, self.lower)
        h_high, s_high, v_high = map(int, self.upper)

        # --- 特殊情況：全0 → 空選取；全滿 → 全選取 ---
        if (h_low, s_low, v_low) == (0, 0, 0) and (h_high, s_high, v_high) == (0, 0, 0):
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        elif (h_low, s_low, v_low) == (0, 0, 0) and (h_high, s_high, v_high) == (H_MAX, S_MAX, V_MAX):
            mask = np.full(hsv.shape[:2], 255, dtype=np.uint8)
        elif (h_low, s_low, v_low) == (H_MAX, S_MAX, V_MAX) and (h_high, s_high, v_high) == (H_MAX, S_MAX, V_MAX):
            # 兼容：若上下限都被拉到最大，也視為全選
            mask = np.full(hsv.shape[:2], 255, dtype=np.uint8)
        else:
            # --- 一般情況：沿用你原本的區間與跨零點邏輯 ---
            if h_low <= h_high:
                mask = cv2.inRange(
                    hsv,
                    (h_low, s_low, v_low),
                    (h_high, s_high, v_high)
                )
            else:
                mask1 = cv2.inRange(hsv, (0, s_low, v_low), (h_high, s_high, v_high))
                mask2 = cv2.inRange(hsv, (h_low, s_low, v_low), (H_MAX, s_high, v_high))
                mask = cv2.bitwise_or(mask1, mask2)

            # 形態學清雜訊（只在一般情況下做，避免把「全選」變稀疏）
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel3, iterations=2)

        # 後處理與發佈：保持你原本的流程
        mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')

        key = f"{self.select_color.capitalize()}Label"
        if key in self.color_labels:
            b, g, r = self.color_labels[key]['color']
            color_img = np.zeros_like(resized)
            color_img[:] = (b, g, r)

            fg = cv2.bitwise_and(color_img, color_img, mask=mask)
            inv = cv2.bitwise_not(mask)
            bg = cv2.bitwise_and(resized, resized, mask=inv)
            composed = cv2.add(bg, fg)

            color_msg = self.bridge.cv2_to_imgmsg(composed, encoding='bgr8')
            self.processed_image.publish(color_msg)
        else:
            self.processed_image.publish(self.bridge.cv2_to_imgmsg(resized, encoding='bgr8'))

        vis_msg = self.bridge.cv2_to_imgmsg(resized, encoding='bgr8')
        return vis_msg, mask_msg



    def build_all_hsv_table(self, hsv, resized, stamp):
        """
        根據 self.HSVColorRange 對每個顏色做 inRange + 形態學清理，
        輸出：
        - /detections/<label>   ：各色精簡 JSON（節流+去抖）
        - /object_info          ：全部顏色的精簡 JSON（節流+去抖）
        - /label_matrix (Image) ：mono8 的 total_mask（影像節流）
        - /build_image (Image)  ：彩色可視化（影像節流）
        參數:
        hsv     : HSV 影像 (H:0~179, S/V:0~255)
        resized : 對應 BGR 影像（用於可視化上色；保留你的變數名稱）
        stamp   : {'sec': int, 'nanosec': int}
        """
        h, w = hsv.shape[:2]
        total_mask = np.zeros((h, w), dtype=np.uint8)     # 純黑白二值化
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)  # BGR 偽彩色輸出

        # 全色偵測結果（精簡）
        detections_all = {label: [] for label in self.HSVColorRange.keys()}

        # === 逐色門檻 + 形態學 + 蒐集結果 ===
        for label, color_obj in self.HSVColorRange.items():
            h_low, h_high = int(color_obj.HueMin), int(color_obj.HueMax)
            s_low, s_high = int(color_obj.SaturationMin), int(color_obj.SaturationMax)
            v_low, v_high = int(color_obj.BrightnessMin), int(color_obj.BrightnessMax)

            # 空設定直接跳過
            if (h_low, h_high, s_low, s_high, v_low, v_high) == (0, 0, 0, 0, 0, 0):
                continue

            # Hue 跨 0 度處理
            if h_low <= h_high:
                mask_i = cv2.inRange(hsv, (h_low, s_low, v_low), (h_high, s_high, v_high))
            else:
                m1 = cv2.inRange(hsv, (0,   s_low, v_low), (h_high, s_high, v_high))
                m2 = cv2.inRange(hsv, (h_low, s_low, v_low), (179,    s_high, v_high))
                mask_i = cv2.bitwise_or(m1, m2)

            # 形態學開運算去雜訊（只建一次的 kernel：self.kernel3）
            mask_i = cv2.morphologyEx(mask_i, cv2.MORPH_OPEN, self.kernel3, iterations=2)

            # 累積總 mask（用清理後的 mask_i）
            total_mask = cv2.bitwise_or(total_mask, mask_i)

            # 偽彩色（用清理後的 mask_i）
            label_key = label.capitalize() + "Label"
            bgr_color = np.array(self.color_labels.get(label_key, {"color": [255, 255, 255]})["color"],
                                dtype=np.uint8)
            color_mask[mask_i > 0] = bgr_color

            # ====== 連通元件（比 findContours 更省） ======
            num, labels_cc, stats, centroids = cv2.connectedComponentsWithStats(mask_i, connectivity=8)
            color_list_compact = []
            for i in range(1, num):  # 0 是背景
                x, y, w_box, h_box, area = stats[i]
                if area < 50:
                    continue
                cx, cy = centroids[i]
                item = {
                    "bbox": (int(x), int(y), int(w_box), int(h_box)),
                    "centroid": (int(cx), int(cy)),
                    "area": float(area),
                    "aspect_ratio": float(w_box) / float(h_box) if h_box > 0 else 0.0,
                    "label": label
                }
                color_list_compact.append(item)
                detections_all[label].append(item)

            # ====== 每色各發一則 JSON（節流 + 去抖） ======
            msg_color = {
                "stamp": stamp,
                "width": w, "height": h,
                "label": label,
                "objects": color_list_compact
            }
            payload = json.dumps(msg_color, separators=(',', ':'))
            now = time.time()
            key = f"det_{label}"
            if payload != self._last_payload[label] and (now - self._last_pub_t[key] >= self._pub_period):
                try:
                    self.det_pubs[label].publish(String(data=payload))
                    self._last_payload[label] = payload
                    self._last_pub_t[key] = now
                except Exception as e:
                    self.get_logger().warning(f"publish detections/{label} failed: {e}")

        # === 發佈總表 object_info（節流 + 去抖） ===
        detections_all["_stamp"] = stamp
        payload_all = json.dumps(detections_all, separators=(',', ':'))
        now = time.time()
        if payload_all != self._last_info_payload and (now - self._last_pub_t["info"] >= self._pub_period):
            try:
                self.info_pub.publish(String(data=payload_all))
                self._last_info_payload = payload_all
                self._last_pub_t["info"] = now
            except Exception as e:
                self.get_logger().warning(f"publish object_info failed: {e}")

        # === 發佈 total_mask 為 mono8 Image（影像節流） ===
        img_msg = self.bridge.cv2_to_imgmsg(total_mask, encoding='mono8')
        try:
            img_msg.header.stamp.sec = int(stamp.get('sec', 0))
            img_msg.header.stamp.nanosec = int(stamp.get('nanosec', 0))
        except Exception:
            pass
        now = time.time()
        if now - self._last_img_pub["mask_image"] >= self._img_period:
            self.label_pub.publish(img_msg)
            self._last_img_pub["mask_image"] = now

        # === 彩色可視化影像（把 color_mask 套 total_mask，影像節流） ===
        vis_all = cv2.bitwise_and(color_mask, color_mask, mask=total_mask)

        # 畫外部請求（line/rect）
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
        try:
            vis_msg_all.header.stamp.sec = int(stamp.get('sec', 0))
            vis_msg_all.header.stamp.nanosec = int(stamp.get('nanosec', 0))
        except Exception:
            pass
        now = time.time()
        if now - self._last_img_pub["build_image"] >= self._img_period:
            self.processed_image.publish(vis_msg_all)
            self._last_img_pub["build_image"] = now


        
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