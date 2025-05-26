#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import os
import sys

def parse_args():
    p = argparse.ArgumentParser(
        description="Headless Camera HSV 顏色分割 → 輸出影片"
    )
    p.add_argument('--hmin', type=int, default=  0, help='H 下界 (0–180)')
    p.add_argument('--smin', type=int, default=  0, help='S 下界 (0–255)')
    p.add_argument('--vmin', type=int, default=  0, help='V 下界 (0–255)')
    p.add_argument('--hmax', type=int, default=180, help='H 上界 (0–180)')
    p.add_argument('--smax', type=int, default=255, help='S 上界 (0–255)')
    p.add_argument('--vmax', type=int, default=255, help='V 上界 (0–255)')
    p.add_argument('--kernel', type=int, default=5,
                   help='形態學運算核大小 (必須為奇數)')
    p.add_argument('--output-video', '-o', required=True,
                   help='輸出影片檔名 (e.g. segmented.avi)')
    return p.parse_args()

def main():
    args = parse_args()

    # 打開預設攝影機
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("❌ 無法開啟攝影機 (index=0)")

    # 讀幀率與解析度
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 影片 Writer：左右拼接原圖+分割結果
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(args.output_video, fourcc, fps, (W*2, H))

    # 形態學核
    k = args.kernel if args.kernel % 2 == 1 else args.kernel + 1
    morph_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    print("開始處理，按 Ctrl+C 停止並寫檔…")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ 讀取影格失敗，退出")
                break

            # BGR → HSV
            hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower = np.array([args.hmin, args.smin, args.vmin])
            upper = np.array([args.hmax, args.smax, args.vmax])

            # 分割
            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  morph_k)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, morph_k)
            result = cv2.bitwise_and(frame, frame, mask=mask)

            # 左右拼接後寫入影片
            combined = np.hstack((frame, result))
            writer.write(combined)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        writer.release()
        print(f"✅ 處理結束，分割影片已存至：{args.output_video}")

if __name__ == '__main__':
    main()
