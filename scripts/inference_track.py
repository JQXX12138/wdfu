"""
目标跟踪脚本 —— 三角洲人物检测
基于 YOLOv8 内置 ByteTrack，不同人物不同颜色框
用法:
    python inference_track.py --source D:/path/to/video.mp4
    python inference_track.py --source D:/path/to/video.mp4 --conf 0.5
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# 复用视频 I/O 和 FPS 模块
from inference_video import open_video, create_writer, FPSMeter, draw_fps


# ── 按 track_id 分配颜色（HSV 色环均匀分布）────────────

def color_by_id(track_id: int) -> tuple:
    """
    传入 track_id，返回 (B, G, R) 颜色元组
    色相间隔 43°，确保相邻 ID 颜色区分明显
    """
    h = (track_id * 43) % 360        # H: 色相
    s = 255                           # S: 饱和度
    v = 255                           # V: 明度
    # HSV → BGR
    hsv = np.array([[[h, s, v]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    b, g, r = int(bgr[0, 0, 0]), int(bgr[0, 0, 1]), int(bgr[0, 0, 2])
    return (b, g, r)


# ── 在帧上绘制跟踪框 ──────────────────────────────────

def draw_tracks(img: np.ndarray, results, class_names: dict) -> np.ndarray:
    """
    在图像上绘制跟踪框：不同 ID 不同颜色，标签 "ID:1 player 0.95"
    如果 results 为空（tracker 未关联），原图返回
    """
    img_out = img.copy()
    if results[0].boxes is None or results[0].boxes.id is None:
        return img_out

    boxes = results[0].boxes.xyxy.cpu().numpy()       # (n, 4) 像素坐标
    confs = results[0].boxes.conf.cpu().numpy()        # (n,)
    clss  = results[0].boxes.cls.cpu().numpy().astype(int)  # (n,)
    ids   = results[0].boxes.id.cpu().numpy().astype(int)   # (n,) track ID

    for (x1, y1, x2, y2), conf, cls_id, track_id in zip(boxes, confs, clss, ids):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        color = color_by_id(track_id)

        # 彩色框
        cv2.rectangle(img_out, (x1, y1), (x2, y2), color, thickness=2)

        # 标签 "ID:1 player 0.95"，放在框顶部中央
        label = f"ID:{track_id} {class_names.get(cls_id, str(cls_id))} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        (tw, th), baseline = cv2.getTextSize(label, font, scale, 1)
        # 文字居中对齐框顶
        text_x = x1 + (x2 - x1 - tw) // 2
        text_x = max(text_x, 0)
        text_y = y1 - 4
        if text_y - th < 0:
            text_y = y1 + th + 4  # 框太靠上，放内部

        # 黑色描边 + 白色字
        cv2.putText(img_out, label, (text_x, text_y), font, scale,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img_out, label, (text_x, text_y), font, scale,
                    (255, 255, 255), 1, cv2.LINE_AA)

    return img_out


# ── 命令行参数 ─────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 目标跟踪")
    parser.add_argument(
        "--source", "-s", type=str, required=True,
        help="视频文件路径",
    )
    parser.add_argument(
        "--weights", "-w", type=str,
        default="D:/delta-force-yolo/models/best.pt",
        help="模型权重路径",
    )
    parser.add_argument(
        "--conf", "-c", type=float, default=0.25,
        help="置信度阈值",
    )
    parser.add_argument(
        "--output", "-o", type=str,
        default="D:/delta-force-yolo/output",
        help="输出目录",
    )
    return parser.parse_args()


# ── 主流程 ─────────────────────────────────────────────

def main():
    args = parse_args()

    # 输入校验
    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"视频文件不存在: {source_path}")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 输出文件名
    out_name = f"{source_path.stem}_track.mp4"
    out_path = str(output_dir / out_name)

    # 加载模型
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"模型权重不存在: {weights_path}")
    model = YOLO(str(weights_path))
    class_names = {0: "player"}
    try:
        class_names = model.names
    except Exception:
        pass

    # 打开视频 + 创建输出写入器
    cap, video_fps, frame_w, frame_h, total_frames = open_video(args.source)
    writer = create_writer(out_path, video_fps, frame_w, frame_h)

    print(f"输出: {out_path}")
    print(f"置信度阈值: {args.conf}")
    print("开始跟踪...")
    print("-" * 50)

    # ── 逐帧跟踪 ──────────────────────────────────
    fps_meter = FPSMeter(window_size=10)
    frame_idx = 0
    total_detections = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # .track() 内置 ByteTrack，persist=True 跨帧保持 ID
        t0 = time.perf_counter()
        results = model.track(
            frame, conf=args.conf, persist=True,
            tracker="bytetrack.yaml", verbose=False,
        )
        infer_ms = (time.perf_counter() - t0) * 1000

        # 统计检测数（可能为 0）
        if results[0].boxes is not None and results[0].boxes.id is not None:
            total_detections += results[0].boxes.shape[0]

        # 更新 FPS
        fps_meter.update(infer_ms)
        current_fps = fps_meter.get_fps()

        # 画跟踪框 + FPS
        frame_out = draw_tracks(frame, results, class_names)
        draw_fps(frame_out, current_fps)

        # 写入输出视频
        writer.write(frame_out)

        # 进度显示
        if frame_idx % 10 == 0 or frame_idx == total_frames:
            percent = frame_idx / total_frames * 100 if total_frames > 0 else 0
            print(f"\r  进度: {frame_idx}/{total_frames} ({percent:.1f}%)  |  "
                  f"推理耗时: {infer_ms:.1f}ms  |  FPS: {current_fps:.1f}", end="")

    # ── 收尾 ──────────────────────────────────────
    print()
    cap.release()
    writer.release()
    print("-" * 50)
    print(f"跟踪完成! 总检测数: {total_detections}")
    print(f"输出视频: {out_path}")


if __name__ == "__main__":
    main()
