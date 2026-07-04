"""
摄像头实时检测脚本 —— 三角洲人物检测
用法:
    python inference_camera.py
    python inference_camera.py --conf 0.5
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

# 必须在 import ultralytics 之前保存原始 cv2.imshow，
# 因为 ultralytics 会 monkey-patch 它，Windows 上可能不兼容。
_orig_imshow = cv2.imshow

from ultralytics import YOLO

# 恢复原始 imshow
cv2.imshow = _orig_imshow

# 复用已有模块
from inference_image import draw_detections
from inference_video import FPSMeter, draw_fps


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 摄像头实时检测")
    parser.add_argument(
        "--weights", "-w", type=str,
        default="D:/delta-force-yolo/models/best.pt",
        help="模型权重路径",
    )
    parser.add_argument(
        "--conf", "-c", type=float, default=0.5,
        help="置信度阈值",
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="摄像头索引（默认 0）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 加载模型 ──────────────────────────────────────
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"模型权重不存在: {weights_path}")
    model = YOLO(str(weights_path))
    class_names = {0: "player"}
    try:
        class_names = model.names
    except Exception:
        pass

    # ── 打开摄像头 ────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"错误: 无法打开摄像头 (index={args.camera})")
        return

    print(f"摄像头已打开 (index={args.camera})")
    print(f"置信度阈值: {args.conf}")
    print("按 Q 键退出")

    fps_meter = FPSMeter(window_size=10)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("警告: 无法读取摄像头画面")
            break

        # 推理计时
        t0 = time.perf_counter()
        results = model(frame, conf=args.conf, verbose=False)
        infer_ms = (time.perf_counter() - t0) * 1000

        # 更新 FPS
        fps_meter.update(infer_ms)
        current_fps = fps_meter.get_fps()

        # 画检测框 + FPS
        frame_out = draw_detections(frame, results, class_names)
        draw_fps(frame_out, current_fps)

        # 左上角提示文字
        cv2.putText(frame_out, "Press Q to quit", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame_out, "Press Q to quit", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

        # 显示
        cv2.imshow("YOLO Camera Detection", frame_out)

        # 按 Q 退出（不区分大小写）
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            print("用户按下 Q 键，退出检测")
            break

    # ── 释放资源 ──────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print("摄像头已释放")


if __name__ == "__main__":
    main()
