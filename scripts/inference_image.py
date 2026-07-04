"""
图片推理脚本 —— 三角洲人物检测
用法:
    python inference_image.py --source D:/path/to/image.jpg
    python inference_image.py --source D:/path/to/folder
    python inference_image.py --source D:/path/to/folder --conf 0.5 --weights D:/delta-force-yolo/models/best.pt
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="YOLOv8 图片推理")
    parser.add_argument(
        "--source", "-s", type=str, required=True,
        help="图片路径或文件夹路径",
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
    parser.add_argument(
        "--line-width", "-lw", type=int, default=2,
        help="框线宽度",
    )
    return parser.parse_args()


def get_image_files(source: str) -> list:
    """获取待推理的图片路径列表"""
    source = Path(source)
    if source.is_file():
        return [source]
    elif source.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        files = [p for p in sorted(source.iterdir()) if p.suffix.lower() in exts]
        if not files:
            raise FileNotFoundError(f"目录中没有图片文件: {source}")
        return files
    else:
        raise FileNotFoundError(f"路径不存在: {source}")


def draw_detections(img: np.ndarray, results, class_names: dict) -> np.ndarray:
    """
    在图像上绘制检测框
    - 红色框 (0, 0, 255)
    - 标签 "player 0.95" 在框左上角，白字 + 黑色描边
    """
    img_out = img.copy()
    if results[0].boxes is None:
        return img_out

    boxes = results[0].boxes.xyxy.cpu().numpy()       # (n, 4) 像素坐标
    confs = results[0].boxes.conf.cpu().numpy()        # (n,)
    clss  = results[0].boxes.cls.cpu().numpy().astype(int)  # (n,)

    for (x1, y1, x2, y2), conf, cls_id in zip(boxes, confs, clss):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        # 红色框
        cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
        # 标签文字 —— 黑色描边 + 白色字
        label = f"{class_names.get(cls_id, str(cls_id))} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        (tw, th), baseline = cv2.getTextSize(label, font, scale, 1)
        # 文字放在框左上角之上，不超出画面
        text_y = y1 - 4
        if text_y - th < 0:
            text_y = y1 + th + 4  # 框太靠上，放框内侧上方
        # 黑色描边
        cv2.putText(img_out, label, (x1, text_y), font, scale, (0, 0, 0), 3, cv2.LINE_AA)
        # 白色字
        cv2.putText(img_out, label, (x1, text_y), font, scale, (255, 255, 255), 1, cv2.LINE_AA)

    return img_out


def inference_image(model: YOLO, img_path: Path, output_dir: Path, conf: float,
                    class_names: dict) -> int:
    """
    对单张图片推理并保存结果
    返回检测到的目标数量
    """
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [跳过] 无法读取图片: {img_path}")
        return 0

    # 推理
    results = model(img, conf=conf, verbose=False)
    num_det = results[0].boxes.shape[0] if results[0].boxes is not None else 0

    if num_det == 0:
        # 无检测目标：原图直接复制
        shutil.copy2(img_path, output_dir / img_path.name)
        print(f"  [无检测目标] {img_path.name}")
    else:
        # 画框 + 保存
        img_ann = draw_detections(img, results, class_names)
        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), img_ann)
        print(f"  [检测到 {num_det} 个目标] {img_path.name} → {out_path}")

    return num_det


def main():
    args = parse_args()

    # ── 加载模型 ──────────────────────────────────────
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"模型权重不存在: {weights_path}")
    model = YOLO(str(weights_path))
    # 从模型读取类别名，兜底用 data.yaml
    class_names = {0: "player"}
    try:
        class_names = model.names
    except Exception:
        pass

    # ── 创建输出目录 ─────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 收集待推理图片 ───────────────────────────────
    img_files = get_image_files(args.source)
    print(f"找到 {len(img_files)} 张图片")
    print(f"模型: {weights_path}")
    print(f"输出: {output_dir}")
    print("-" * 50)

    # ── 逐张推理 ─────────────────────────────────────
    total_detections = 0
    for img_path in img_files:
        n = inference_image(model, img_path, output_dir, args.conf, class_names)
        total_detections += n

    # ── 汇总 ─────────────────────────────────────────
    print("-" * 50)
    print(f"处理完成: {len(img_files)} 张图片, 共 {total_detections} 个检测目标")


if __name__ == "__main__":
    main()
