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
            "RedLabel":     {"label": 5, "color": [  0, 255, 255] },   # 黃
            "YellowLabel":  {"label": 6, "color": [128, 128,   0] },   # 黃綠
            "WhiteLabel":   {"label": 7, "color": [  0, 255, 255] },   # 青綠
            "OthersLabel":  {"label": 8, "color": [255,   0, 128] },   # 紫粉
        }
        # Publisher: 物件資訊 (JSON)
        self.info_pub = self.create_publisher(String, 'object_info', 10)
        # ↓↓↓ 這裡重建一個跟 HSVColorRange key 一樣的小寫 map ↓↓↓
        import numpy as np
        self.raw_HSVColorRange = {
            label: np.array(
                self.color_labels[f"{label.capitalize()}Label"]["color"],
                dtype=np.uint8
            )
            for label in self.labels
        }
        self.HSVColorRange = {label: ColorRange(LabelName=label) for label in self.labels}
        self.raw_HSVColorRange = {label: ColorRange(LabelName=label) for label in self.color_labels}
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
        # HSV 查詢表
        # self.table_bmp_to_hsv = self.build_hsv_table()

        # 顏色分類的樣本和標籤
        self.bmp_sample = np.random.randint(0, 9, size=(64 * 64 * 64,), dtype=np.uint8)


        
                # 如果 ColorDeep=256，就跑 0–255
        self.ColorDeep = 256  
        # 預先分配一個 (R, G, B, HSV) 的查表陣列
        # dtype=uint8：讓 H、S、V 都落在 0–255 之間
        # self.TableBMPtoHSV = np.zeros(
        #     (self.ColorDeep, self.ColorDeep, self.ColorDeep, 3),
        #     dtype=np.uint8
        # )
        self.BuiltTable = False
    #

    ##############################  load hsv  ########################
    def load_hsv_info_callback(self, request, response):
        print("Loading HSV data from INI file...")
        # """強制從 .ini 更新後，再回應 HSV 參數"""
        # print(self.HSVColorRange)  # Debug
        self.select_color = request.colorlabel
        color_data = self.HSVColorRange.get(request.colorlabel)
        # print(f"Retrieved color_data: {color_data}")

        if color_data:
            # 這裡 **只在回應時乘 360 或 100**，而不會影響 `self.HSVColorRange` 內的數據
            response.hmin = int(color_data.HueMin)  # **確保只乘一次**
            response.hmax = int(color_data.HueMax)
            response.smin = int(color_data.SaturationMin)
            response.smax = int(color_data.SaturationMax)
            response.vmin = int(color_data.BrightnessMin)
            response.vmax = int(color_data.BrightnessMax)
            self.lower = np.array([response.hmin, response.smin, response.vmin], dtype=np.uint8)
            self.upper = np.array([response.hmax, response.smax, response.vmax], dtype=np.uint8)
            print(f"Returning response: hmin={response.hmin}, hmax={response.hmax}, "
                f"smin={response.smin}, smax={response.smax}, vmin={response.vmin}, vmax={response.vmax}")
        else:
            print(f"Warning: Color label '{request.colorlabel}' not found!")
        return response

    # def color_model_HSV_callback(self, msg):
    #     """更新並儲存 HSV 原始數據"""
    #     self.get_logger().info(f"Received HSV Value: {msg}")
    #     # `self.HSVColorRange` 是「已計算後的數據」
    #     self.HSVColorRange[self.select_color].HueMax = msg.hmax
    #     self.HSVColorRange[self.select_color].HueMin = msg.hmin
    #     self.HSVColorRange[self.select_color].SaturationMax = msg.smax
    #     self.HSVColorRange[self.select_color].SaturationMin = msg.smin
    #     self.HSVColorRange[self.select_color].BrightnessMax = msg.vmax
    #     self.HSVColorRange[self.select_color].BrightnessMin = msg.vmin

    #     print(f"Updated raw HSV data for {self.select_color}: {self.raw_HSVColorRange[self.select_color]}")
    #     print(f"Updated computed HSV data for {self.select_color}: {self.HSVColorRange[self.select_color]}")

    def color_model_HSV_callback(self, msg):
        self.get_logger().info(f"Received HSV Value: {msg}")

        # 更新 `self.HSVColorRange` (原始存儲格式)
        self.HSVColorRange[self.select_color].HueMax = msg.hmax
        self.HSVColorRange[self.select_color].HueMin = msg.hmin
        self.HSVColorRange[self.select_color].SaturationMax = msg.smax
        self.HSVColorRange[self.select_color].SaturationMin = msg.smin
        self.HSVColorRange[self.select_color].BrightnessMax = msg.vmax
        self.HSVColorRange[self.select_color].BrightnessMin = msg.vm你要
        self.lower = np.array([msg.hmin, msg.smin, msg.vmin], dtype=np.uint8)
        self.upper = np.array([msg.hmax, msg.smax, msg.vmax], dtype=np.uint8)
        # self.HSV_BuildingTable(self.HSVColorRange)

        print(f"Updated computed HSV data for {self.select_color}: {self.HSVColorRange[self.select_color]}")
 
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

    # def image_callback(self, msg: Image):
    #     try:
    #         # 1) ROS Image → OpenCV
    #         cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    #         # 2) Resize & HSV
    #         resized = cv2.resize(cv_img, (320, 240))
    #         self.hsv     = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    #         if self.lower is not None and self.upper is not None:

    #         # # 4) 做 inRange → 二值化
    #             mask   = cv2.inRange(self.hsv , self.lower, self.upper)  # 在範圍內255，否則0

    #         # 5) Publish 黑白圖
    #             bin_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
    #             self.processed_image.publish(bin_msg)

    #     except Exception as e:
    #         self.get_logger().error(f"Failed to process image: {e}")


    # def image_callback(self, msg: Image):
    #     try:
    #         # 1) ROS Image → OpenCV
    #         cv_img  = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    #         # 2) Resize & HSV
    #         resized = cv2.resize(cv_img, (320, 240))
    #         self.hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    #         if self.lower is None or self.upper is None:
    #             return

    #         # 3) 黑白二值化
    #         mask = cv2.inRange(self.hsv, self.lower, self.upper)  # 0 or 255

  
    #         # 4) 用驼峰Label做索引
    #         key = f"{self.select_color.capitalize()}Label"
    #         if key not in self.color_labels:
    #             self.get_logger().warn(f"No color defined for {self.select_color}")
    #             return
    #         b, g, r = self.color_labels[key]["color"]

    #         # 5) 建一张纯色BGR图
    #         color_img = np.zeros_like(resized)
    #         color_img[:] = (b, g, r)

    #         # 6) 把 mask 套到純色圖：mask==255 的地方保留顏色，其他變黑
    #         colored_mask = cv2.bitwise_and(color_img, color_img, mask=mask)

    #         # 7) publish
    #         out_msg = self.bridge.cv2_to_imgmsg(colored_mask, encoding='bgr8')
    #         self.processed_image.publish(out_msg)

    #     except Exception as e:
    #         self.get_logger().error(f"Failed to process image: {e}")


    # def image_callback(self, msg: Image):
    #     try:
    #         # 1) ROS Image → OpenCV
    #         cv_img  = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    #         # 2) Resize & HSV
    #         resized = cv2.resize(cv_img, (320, 240))
    #         self.hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    #         if self.lower is None or self.upper is None:
    #             return

    #         # 3) 黑白二值化
    #         mask = cv2.inRange(self.hsv, self.lower, self.upper)  # 0 or 255

    #         # ── 新增：形态学处理 ────────────────────────────────
    #         # 用一个 5x5 的椭圆核，先闭运算（填充小洞），再开运算（去掉小噪点）
    #         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    #         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    #         # 如果只想单独做腐蚀或膨胀，也可以：
    #         mask = cv2.erode(mask,  kernel, iterations=1)
    #         mask = cv2.dilate(mask, kernel, iterations=2)
    #         # ── 形态学处理结束 ────────────────────────────────

    #         # 4) 用驼峰Label做索引，拿到 BGR 颜色
    #         key = f"{self.select_color.capitalize()}Label"
    #         if key not in self.color_labels:
    #             self.get_logger().warn(f"No color defined for {self.select_color}")
    #             return
    #         b, g, r = self.color_labels[key]["color"]

    #         # 5) 建一張純色 BGR 圖
    #         color_img = np.zeros_like(resized)
    #         color_img[:] = (b, g, r)

    #         # 6) 把 mask 套到純色圖：mask==255 的地方保留顏色，其他變黑
    #         colored_mask = cv2.bitwise_and(color_img, color_img, mask=mask)

    #         # 7) publish
    #         out_msg = self.bridge.cv2_to_imgmsg(colored_mask, encoding='bgr8')
    #         self.processed_image.publish(out_msg)

    #     except Exception as e:
    #         self.get_logger().error(f"Failed to process image: {e}")

    # def image_callback(self, msg: Image):
    #     try:
    #         # 1) ROS Image → OpenCV
    #         cv_img  = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    #         # 2) Resize & HSV
    #         resized = cv2.resize(cv_img, (320, 240))
    #         hsv     = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    #         # 3) 必须先有 lower/upper
    #         if self.lower is None or self.upper is None:
    #             return

    #         # 4) 黑白二值化
    #         mask = cv2.inRange(hsv, self.lower, self.upper)  # 0 or 255

    #         # 5) 形态学处理：闭运算填洞 + 开运算去噪 + 单独腐蚀/膨胀
    #         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    #         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    #         mask = cv2.erode(mask,  kernel, iterations=1)
    #         mask = cv2.dilate(mask, kernel, iterations=2)
    #         # print(type(mask))
    #         mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
    #         print(type(mask_msg))
    #         self.mask_pub.publish(mask_msg) 
    #         # 6) 物件检测：连通域 + 统计
    #         num_labels, labels_img, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    #         objects = []
    #         for i in range(1, num_labels):  # 跳过 background(0)
    #             x, y, w, h, area = stats[i]
    #             cx, cy          = centroids[i]
    #             objects.append({
    #                 'id':       i,
    #                 'bbox':     (x, y, w, h),
    #                 'area':     int(area),
    #                 'centroid': (float(cx), float(cy)),
    #             })
    #             # 在 resized 上画框和重心
    #             cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #             # cv2.circle(resized, (int(cx), int(cy)), 4, (255, 0, 0), -1)
            

    #         # 7) log 或存起来给下游用
    #         # self.get_logger().info(f"Detected {len(objects)} objects: {objects}")
    #         self.last_objects = objects  # 如果后面要用

    #         # 8) 根据 select_color 拿 BGR，并做彩色遮罩
    #         key = f"{self.select_color.capitalize()}Label"
    #         if key not in self.color_labels:
    #             self.get_logger().warn(f"No color defined for {self.select_color}")
    #             return
    #         b, g, r = self.color_labels[key]["color"]
    #         color_img = np.zeros_like(resized)
    #         color_img[:] = (b, g, r)
    #         colored_mask = cv2.bitwise_and(color_img, color_img, mask=mask)

    #         # 9) Publish：彩色遮罩
    #         color_msg = self.bridge.cv2_to_imgmsg(colored_mask, encoding='bgr8')
    #         self.processed_image.publish(color_msg)

    #         # 10) Publish：带框的可视化结果
    #         vis_msg = self.bridge.cv2_to_imgmsg(resized, encoding='bgr8')
    #         self.build_image.publish(vis_msg)

    #     except Exception as e:
    #         self.get_logger().error(f"Failed to process image: {e}")


    def image_callback(self, msg: Image):
        try:
            # 1) ROS Image → OpenCV
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # 2) Resize & HSV
            resized = cv2.resize(cv_img, (320, 240))
            hsv     = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

            # 3) 必须先有 lower/upper
            if self.lower is None or self.upper is None:
                return

            # 4) 黑白二值化
            mask = cv2.inRange(hsv, self.lower, self.upper)

            # 5) 形态学处理
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
            mask = cv2.erode(mask,  kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)
            mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
            print(type(mask_msg))
            self.mask_pub.publish(mask_msg) 
            # 6) 物件检测：连通域 + 统计
            num_labels, labels_img, stats, centroids = \
                cv2.connectedComponentsWithStats(mask, connectivity=8)
            objects = []
            for i in range(1, num_labels):  # 跳过 background(0)
                x, y, w, h, area = stats[i]
                cx, cy = centroids[i]
                objects.append({
                    'id':       i,
                    'bbox':     [int(x), int(y), int(w), int(h)],
                    'area':     int(area),
                    'centroid': [float(cx), float(cy)]
                })
                # 可視化框
                cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 7) Publish 物件資訊 (JSON)
            info_msg = String()
            info_msg.data = json.dumps(objects)
            self.info_pub.publish(info_msg)

            # 8) 彩色遮罩
            key = f"{self.select_color.capitalize()}Label"
            if key in self.color_labels:
                b, g, r = self.color_labels[key]['color']
                color_img = np.zeros_like(resized)
                color_img[:] = (b, g, r)
                colored_mask = cv2.bitwise_and(color_img, color_img, mask=mask)
                color_msg = self.bridge.cv2_to_imgmsg(colored_mask, encoding='bgr8')
                self.processed_image.publish(color_msg)

            # 9) Publish 帶框可視化影像
            vis_msg = self.bridge.cv2_to_imgmsg(resized, encoding='bgr8')
            self.build_image.publish(vis_msg)

        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

    # def image_callback(self, msg: Image):
    #     try:
    #         # 1) ROS Image → OpenCV BGR
    #         cv_img  = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
    #         # 2) Resize & HSV
    #         resized = cv2.resize(cv_img, (320, 240))
    #         hsv     = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    #         # --- 單色檢測 (select_color) ---
    #         if self.lower is not None and self.upper is not None:
    #             mask_sel = cv2.inRange(hsv, self.lower, self.upper)
    #             key = f"{self.select_color.capitalize()}Label"
    #             if key in self.color_labels:
    #                 b,g,r      = self.color_labels[key]['color']
    #                 col        = np.zeros_like(resized)
    #                 col[:]     = (b,g,r)
    #                 sel_col    = cv2.bitwise_and(col, col, mask=mask_sel)
    #                 out_sel    = self.bridge.cv2_to_imgmsg(sel_col, encoding='bgr8')
    #                 self.processed_image.publish(out_sel)

    #                 # 單色物件資訊
    #                 n, lbl, stats, cents = cv2.connectedComponentsWithStats(mask_sel)
    #                 objs_sel = []
    #                 for i in range(1, n):
    #                     x,y,w,h,area = stats[i]
    #                     cx,cy        = cents[i]
    #                     objs_sel.append({
    #                         'id':       i,
    #                         'bbox':     [int(x),int(y),int(w),int(h)],
    #                         'area':     int(area),
    #                         'centroid': [float(cx),float(cy)]
    #                     })
    #                 sel_msg = String()
    #                 sel_msg.data = json.dumps(objs_sel)
    #                 # self.info_pub.publish(sel_msg)
    #         # --- 單色檢測結束 ---

    #         # --- 全部顏色合成檢測 ---
    #         combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    #         for lbl_name, info in self.color_labels.items():
    #             cr = self.HSVColorRange[lbl_name.replace('Label','').lower()]
    #             low = np.array([int(cr.HueMin*179), int(cr.SaturationMin*255), int(cr.BrightnessMin*255)], np.uint8)
    #             high= np.array([int(cr.HueMax*179), int(cr.SaturationMax*255), int(cr.BrightnessMax*255)], np.uint8)
    #             m   = cv2.inRange(hsv, low, high)
    #             k   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    #             m   = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    #             m   = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k)
    #             combined_mask = cv2.bitwise_or(combined_mask, m)

    #         # 全部物件資訊
    #         n2, lbl2, stats2, cents2 = cv2.connectedComponentsWithStats(combined_mask)
    #         objs_all = []
    #         for i in range(1, n2):
    #             x,y,w,h,area = stats2[i]
    #             cx,cy        = cents2[i]
    #             objs_all.append({
    #                 'id':       i,
    #                 'bbox':     [int(x),int(y),int(w),int(h)],
    #                 'area':     int(area),
    #                 'centroid': [float(cx),float(cy)]
    #             })
    #         all_msg = String()
    #         all_msg.data = json.dumps(objs_all)
    #         self.info_pub.publish(all_msg)

    #         # 發布合成二值化影像
    #         comb_msg = self.bridge.cv2_to_imgmsg(combined_mask, encoding='mono8')
    #         self.build_image.publish(comb_msg)

    #         # 可視化帶框 (全部)
    #         vis = resized.copy()
    #         for obj in objs_all:
    #             x,y,w,h = obj['bbox']
    #             cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,0), 2)
    #         vis_msg = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
    #         # self.visualized_pub.publish(vis_msg)

    #     except Exception as e:
    #         self.get_logger().error(f"Failed to process image: {e}")


    def location_callback(self, msg):
        """讀取 HSV 參數，更新顏色範圍"""
        print("/////////////////////////////////////////////////")
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

        print("HSVColorRange 更新完畢：", self.HSVColorRange)

    def image_callback_build(self, msg):
        """ROS2 訂閱攝影機影像，並進行分類"""
        try:
            # 轉換 ROS 影像為 OpenCV 格式
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # 縮放影像
            self.build_imageprocess = cv2.resize(self.cv_image, (320, 240))
        except Exception as e:
            print(f"Error in image_callback_build: {e}")

    # def ChangeToColorModel(self, colormodel):
    #     # 將影像分類並生成新影像和標籤模型
    #     image_height, image_width, _ = colormodel.shape
    #     b, g, r = cv2.split(colormodel)
    #     index = (r * 64 * 64 + g * 64 + b).astype(np.int32)
    #     tmp_values = self.bmp_sample[index]

    #     result_image = np.zeros_like(colormodel)
    #     label_model = np.zeros((image_height, image_width), dtype=np.uint8)

    #     for _, data in self.color_labels.items():
    #         mask = (tmp_values == data["label"])
    #         result_image[mask] = data["color"]
    #         label_model[mask] = data["label"]

    #     mask_others = (tmp_values == 0)
    #     result_image[mask_others] = [0, 0, 0]
    #     label_model[mask_others] = 0x00

    #     return result_image, label_model

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
