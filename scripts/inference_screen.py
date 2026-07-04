"""
游戏截屏 + 自动跟枪 —— 三角洲人物检测
用法:
    python inference_screen.py
    python inference_screen.py --conf 0.35 --aim-key capslock
"""

import argparse
import ctypes
import threading
import time
from pathlib import Path

import cv2
import numpy as np

_orig_imshow = cv2.imshow

import mss
from ultralytics import YOLO

cv2.imshow = _orig_imshow

from inference_image import draw_detections
from inference_video import FPSMeter, draw_fps

# ── Windows 鼠标控制（ctypes，无需额外依赖）─────────────

# 虚拟键码，用于监听按键状态
VK_CODES = {
    "capslock": 0x14,
    "ralt": 0xA5,
    "lalt": 0xA4,
    "rshift": 0xA1,
    "lshift": 0xA0,
    "rctrl": 0xA3,
    "lctrl": 0xA2,
    "alt": 0x12,
    "ctrl": 0x11,
    "shift": 0x10,
    "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73,
}


def is_key_pressed(vk_code: int) -> bool:
    """检测按键是否正在按下（高位为 1 表示按下）"""
    return ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x8000 != 0


def move_mouse_relative(dx: int, dy: int):
    """
    鼠标相对移动（Raw Input 兼容）
    SendInput 是游戏认可的标准输入方式
    """
    # MOUSEEVENTF_MOVE = 0x0001
    ctypes.windll.user32.mouse_event(0x0001, dx, dy, 0, 0)


# ── 跟枪线程：持续高频移动鼠标，不受检测帧率限制 ──────

class AimThread:
    """
    独立线程持续跟枪，500Hz 鼠标更新，不受检测 FPS 影响
    """

    def __init__(self, screen_w: int, screen_h: int,
                 dead_zone: int = 3, smooth: float = 0.6):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.dead_zone = dead_zone
        self.smooth = smooth

        self._target = None       # (x1, y1, x2, y2) 由主线程更新
        self._aim_active = False  # 由主线程更新
        self._running = True
        self._lock = threading.Lock()

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def update_target(self, target, aim_active: bool):
        """主线程调用，更新当前跟枪目标和状态"""
        with self._lock:
            self._target = target
            self._aim_active = aim_active

    def stop(self):
        self._running = False

    def _run(self):
        """跟枪线程：每 2ms 读取目标并移动鼠标"""
        while self._running:
            with self._lock:
                target = self._target
                active = self._aim_active

            if active and target is not None:
                x1, y1, x2, y2 = target
                target_cx = (x1 + x2) / 2
                target_cy = (y1 + y2) / 2

                dx = target_cx - self.screen_w / 2
                dy = target_cy - self.screen_h / 2

                if abs(dx) > self.dead_zone or abs(dy) > self.dead_zone:
                    move_mouse_relative(
                        int(dx * self.smooth / 3),
                        int(dy * self.smooth / 3),
                    )

            time.sleep(0.002)  # 500Hz


# ── 目标选择 ──────────────────────────────────────────

def select_target(boxes_xyxy, screen_w: int, screen_h: int,
                  center_weight: float = 0.85, size_weight: float = 0.15):
    """
    从所有检测框中选出最优跟枪目标
    评分 = center_weight × 距离中心近 + size_weight × 框面积大
    返回 (x1, y1, x2, y2) 或 None
    """
    if len(boxes_xyxy) == 0:
        return None

    cx_screen, cy_screen = screen_w / 2, screen_h / 2
    best_score = -1
    best_box = None

    for box in boxes_xyxy:
        x1, y1, x2, y2 = box
        box_cx = (x1 + x2) / 2
        box_cy = (y1 + y2) / 2
        box_w = x2 - x1
        box_h = y2 - y1

        # 离屏幕中心越近越好
        dist = np.sqrt((box_cx - cx_screen) ** 2 + (box_cy - cy_screen) ** 2)
        max_dist = np.sqrt(cx_screen ** 2 + cy_screen ** 2)
        center_score = 1.0 - dist / max_dist

        # 框越大越好（大目标 = 近处敌人，优先级高）
        size_score = min((box_w * box_h) / (screen_w * screen_h * 0.05), 1.0)

        score = center_weight * center_score + size_weight * size_score
        if score > best_score:
            best_score = score
            best_box = box

    return best_box


# ── 参数解析 ───────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 截屏 + 跟枪")
    parser.add_argument("--weights", "-w", type=str,
                        default="D:/delta-force-yolo/models/best.pt")
    parser.add_argument("--conf", "-c", type=float, default=0.35,
                        help="检测置信度阈值")
    parser.add_argument("--display-scale", "-d", type=float, default=0.25,
                        help="全屏模式缩放比例（默认 0.25）")
    parser.add_argument("--display-mode", type=str, default="full",
                        choices=["full", "hud", "off"],
                        help="显示模式: full=缩略检测画面, hud=状态条, off=纯后台")
    parser.add_argument("--aim-key", "-k", type=str, default="ralt",
                        choices=list(VK_CODES.keys()),
                        help="按住该键启动跟枪（默认 ralt=右Alt）")
    parser.add_argument("--dead-zone", type=int, default=8,
                        help="鼠标死区像素（默认 8）")
    parser.add_argument("--smooth", type=float, default=0.5,
                        help="跟枪平滑系数 0~1（默认 0.5）")
    return parser.parse_args()


# ── 主流程 ─────────────────────────────────────────────

def main():
    args = parse_args()

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

    # 截屏器
    sct = mss.mss()
    monitor = sct.monitors[1]
    screen_w = monitor["width"]
    screen_h = monitor["height"]

    # 跟枪线程（独立 500Hz 移动鼠标，不受检测帧率影响）
    aim_key_vk = VK_CODES[args.aim_key]
    aim_thr = AimThread(screen_w, screen_h,
                        dead_zone=args.dead_zone, smooth=args.smooth)

    print(f"屏幕: {screen_w}x{screen_h}")
    print(f"显示模式: {args.display_mode}")
    print(f"跟枪按键: 按住 [{args.aim_key.upper()}] 启动")
    print(f"死区: {args.dead_zone}px  平滑: {args.smooth}")
    print(f"置信度: {args.conf}")
    print("按 Q 退出, F 键降低置信度阈值")
    print("-" * 40)

    # 显示窗口
    win_name = "YOLO Screen Detection"
    if args.display_mode != "off":
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)

    fps_meter = FPSMeter(window_size=10)
    aim_active_prev = False

    try:
        while True:
            # ── 截屏 ──────────────────────────────────
            img = np.array(sct.grab(monitor))
            frame = img[:, :, :3]

            # ── 推理 ──────────────────────────────────
            t0 = time.perf_counter()
            results = model(frame, conf=args.conf, verbose=False)
            infer_ms = (time.perf_counter() - t0) * 1000
            fps_meter.update(infer_ms)

            # ── 检测结果 ──────────────────────────────
            if results[0].boxes is not None:
                all_boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
            else:
                all_boxes = np.array([])

            # ── 跟枪逻辑 ──────────────────────────────
            aim_active = is_key_pressed(aim_key_vk)
            target = select_target(all_boxes, screen_w, screen_h)

            if aim_active and not aim_active_prev:
                print("跟枪: ON")
            elif not aim_active and aim_active_prev:
                print("跟枪: OFF")
            aim_active_prev = aim_active

            # 更新跟枪线程目标（线程自己会高频移动鼠标）
            aim_thr.update_target(target if aim_active else None, aim_active)

            # ── 显示 ──────────────────────────────────
            if args.display_mode == "full":
                # 缩略检测画面：画框 + 缩小显示，小到不会被截屏套娃
                frame_out = draw_detections(frame, results, class_names)
                if target is not None:
                    x1, y1, x2, y2 = map(int, target)
                    color = (0, 255, 0) if aim_active else (255, 165, 0)
                    cv2.rectangle(frame_out, (x1, y1), (x2, y2), color, 3)
                    cv2.line(frame_out,
                             ((x1 + x2) // 2, (y1 + y2) // 2),
                             (screen_w // 2, screen_h // 2), color, 1)
                draw_fps(frame_out, fps_meter.get_fps())
                status = f"AIM: {'ON' if aim_active else 'OFF'}"
                s_color = (0, 255, 0) if aim_active else (0, 0, 255)
                cv2.putText(frame_out, status, (12, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame_out, status, (12, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, s_color, 1, cv2.LINE_AA)
                new_w = int(screen_w * args.display_scale)
                new_h = int(screen_h * args.display_scale)
                frame_show = cv2.resize(frame_out, (new_w, new_h))
                cv2.imshow(win_name, frame_show)

            elif args.display_mode == "hud":
                # 紧凑状态条
                num_det = len(all_boxes)
                fps_val = fps_meter.get_fps()
                bg_color = (30, 60, 30) if aim_active else (40, 40, 40)
                hud = np.full((80, 360, 3), bg_color, dtype=np.uint8)
                cv2.putText(hud, f"FPS: {fps_val:.0f}", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(hud, f"FPS: {fps_val:.0f}", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(hud, f"DET: {num_det}", (120, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(hud, f"DET: {num_det}", (120, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
                aim_text = f"AIM: {'ON' if aim_active else 'OFF'}"
                aim_color = (0, 255, 0) if aim_active else (0, 0, 255)
                cv2.putText(hud, aim_text, (12, 62),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(hud, aim_text, (12, 62),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, aim_color, 1, cv2.LINE_AA)
                cv2.putText(hud, f"hold [{args.aim_key.upper()}] | Q:quit", (160, 62),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(hud, f"hold [{args.aim_key.upper()}] | Q:quit", (160, 62),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
                if target is not None:
                    cv2.circle(hud, (340, 40), 8, aim_color, 2)
                cv2.imshow(win_name, hud)

            # ── 退出 ──────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                print("Q 键退出")
                break
            if key == ord("f") or key == ord("F"):
                # F 键：降低一半置信度阈值
                args.conf = max(0.05, args.conf / 2)
                print(f"置信度阈值: {args.conf:.3f}")

    except KeyboardInterrupt:
        print("Ctrl+C 退出")
    finally:
        aim_thr.stop()
        cv2.destroyAllWindows()
        print("已退出")


if __name__ == "__main__":
    main()
