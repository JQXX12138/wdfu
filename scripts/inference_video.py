"""
视频推理脚本 —— 三角洲人物检测
用法:
    python inference_video.py --source D:/path/to/video.mp4
    python inference_video.py --source D:/path/to/video.mp4 --conf 0.5
"""

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ── 视频 I/O ────────────────────────────────────────────

def open_video(source: str) -> tuple:
    """
    打开视频文件，返回 (cap, fps, frame_w, frame_h, total_frames)
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"视频信息: {frame_w}x{frame_h}, {fps:.2f} FPS, {total_frames} 帧")
    return cap, fps, frame_w, frame_h, total_frames


def create_writer(output_path: str, fps: float, frame_w: int, frame_h: int) -> cv2.VideoWriter:
    """
    创建视频写入器，编码 mp4v
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建输出视频: {output_path}")
    return writer


# ── FPS 计算（平滑平均）─────────────────────────────────

class FPSMeter:
    """
    滑动窗口 FPS 计算器：取最近 N 帧推理时间的平均来算 FPS
    """
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.infer_times = deque(maxlen=window_size)

    def update(self, infer_ms: float):
        """记录一次推理耗时（毫秒）"""
        self.infer_times.append(infer_ms)

    def get_fps(self) -> float:
        """返回平滑后的 FPS"""
        if not self.infer_times:
            return 0.0
        avg_ms = sum(self.infer_times) / len(self.infer_times)
        return 1000.0 / avg_ms if avg_ms > 0 else 0.0


# ── 绘制 FPS 到画面右上角 ───────────────────────────────

def draw_fps(img: np.ndarray, fps: float) -> None:
    """
    在画面右上角绘制 FPS 文字，白字 + 黑色描边
    """
    text = f"FPS: {fps:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    x = img.shape[1] - tw - 12
    y = th + 12
    # 黑色描边
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), 3, cv2.LINE_AA)
    # 白色字
    cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), 1, cv2.LINE_AA)


# ── 复用图片推理的画框函数 ─────────────────────────────

from inference_image import draw_detections


# ── 命令行参数解析 ─────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 视频推理")
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

    # 输出文件名 = 原文件名_result.mp4
    out_name = f"{source_path.stem}_result.mp4"
    out_path = output_dir / out_name

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
    writer = create_writer(str(out_path), video_fps, frame_w, frame_h)

    print(f"输出: {out_path}")
    print(f"置信度阈值: {args.conf}")
    print("开始推理...")
    print("-" * 50)

    # ── 逐帧推理 ──────────────────────────────────
    fps_meter = FPSMeter(window_size=10)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # 推理（计时）
        t0 = time.perf_counter()
        results = model(frame, conf=args.conf, verbose=False)
        infer_ms = (time.perf_counter() - t0) * 1000

        # 更新 FPS
        fps_meter.update(infer_ms)
        current_fps = fps_meter.get_fps()

        # 画检测框 + FPS
        frame_out = draw_detections(frame, results, class_names)
        draw_fps(frame_out, current_fps)

        # 写入输出视频
        writer.write(frame_out)

        # 进度显示（每 10 帧打印一次，避免刷屏）
        if frame_idx % 10 == 0 or frame_idx == total_frames:
            percent = frame_idx / total_frames * 100 if total_frames > 0 else 0
            print(f"\r  进度: {frame_idx}/{total_frames} ({percent:.1f}%)  |  "
                  f"推理耗时: {infer_ms:.1f}ms  |  FPS: {current_fps:.1f}", end="")

    # ── 收尾 ──────────────────────────────────────
    print()  # 换行
    cap.release()
    writer.release()
    print("-" * 50)
    print(f"推理完成! 输出视频: {out_path}")


if __name__ == "__main__":
    main()
