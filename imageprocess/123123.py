import numpy as np
import cv2

class HSVColorProcessor:
    def __init__(self):
        """
        初始化：建立 RGB → HSV 查找表 (LUT)
        """
        self.hsv_lookup_table = self.create_hsv_lookup_table()
        self.color_labels = {
            "Yellow": (30, 60, 0.3, 1.0, 0.5, 1.0),
            "Blue": (85, 130, 0.3, 1.0, 0.3, 1.0),
            "Orange": (10, 25, 0.5, 1.0, 0.5, 1.0),
            "Green": (35, 85, 0.3, 1.0, 0.3, 1.0),
            "White": (0, 180, 0.0, 0.3, 0.8, 1.0),
            "Black": (0, 180, 0.0, 1.0, 0.0, 0.2),
            "Red": ((0, 10), (170, 180), 0.3, 1.0, 0.3, 1.0),  # 紅色分成兩段
            "Others": (0, 180, 0, 1, 0, 1)  # 其他顏色
        }

    def create_hsv_lookup_table(self):
        """
        建立 RGB → HSV 的查找表 (LUT)
        """
        hsv_lookup_table = np.zeros((256, 256, 256, 3), dtype=np.uint8)

        for r in range(256):
            for g in range(256):
                for b in range(256):
                    hsv = cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_BGR2HSV)
                    hsv_lookup_table[r, g, b] = hsv[0, 0]  # 存 H, S, V

        print("✅ HSV Lookup Table 建立完成！")
        return hsv_lookup_table

    def classify_pixel(self, h, s, v):
        """
        根據 HSV 範圍將像素分類到顏色標籤
        """
        for label, ranges in self.color_labels.items():
            if label == "Red":  # 特別處理紅色，因為它分成兩個範圍
                (h1_min, h1_max), (h2_min, h2_max), s_min, s_max, v_min, v_max = ranges
                if (h1_min <= h <= h1_max or h2_min <= h <= h2_max) and s_min <= s <= s_max and v_min <= v <= v_max:
                    return label
            else:
                h_min, h_max, s_min, s_max, v_min, v_max = ranges
                if h_min <= h <= h_max and s_min <= s <= s_max and v_min <= v <= v_max:
                    return label
        return "Others"

    def process_image(self, image):
        """
        處理影像：
        1. 透過查表 (LUT) 取得 HSV 值
        2. 根據 HSV 範圍分類
        3. 產生標籤影像
        """
        height, width, _ = image.shape
        labeled_image = np.zeros((height, width, 3), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                r, g, b = image[y, x]
                h, s, v = self.hsv_lookup_table[r, g, b]  # 查表獲取 HSV
                h = h / 180.0  # 正規化
                s = s / 255.0
                v = v / 255.0
                color_label = self.classify_pixel(h * 180, s, v)

                # 給予顏色標籤 (對應的 RGB)
                if color_label in self.color_labels:
                    labeled_image[y, x] = [r, g, b]  # 保持原色
                else:
                    labeled_image[y, x] = [0, 0, 0]  # 其他變黑

        return labeled_image

# 測試程式
if __name__ == "__main__":
    # ✅ 初始化 HSV 處理器 (建立查找表)
    processor = HSVColorProcessor()

    # ✅ 讀取影像
    image = cv2.imread("image.jpg")

    # ✅ 進行顏色分類
    classified_image = processor.process_image(image)

    # ✅ 顯示結果
    cv2.imshow("Original Image", image)
    cv2.imshow("Classified Image", classified_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
