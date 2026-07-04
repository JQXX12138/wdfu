"""
YOLOv8n 训练脚本 —— 三角洲人物检测
数据集: 单类别 "player", train 2404 / val 291 / test 446
"""

from ultralytics import YOLO  
from pathlib import Path

# ── 路径配置 ────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "data.yaml"

if __name__ == "__main__":
    # ── 加载预训练模型 ──────────────────────────────────
    model = YOLO("yolov8n.pt")

    # ── 训练 ────────────────────────────────────────────
    results = model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=640,
        # 数据增强 —— YOLOv8 默认已开启 mosaic/hsv/scale/fliplr
        # 显式指定确保生效，关闭不想要的增强
        mosaic=1.0,          # 马赛克增强
        hsv_h=0.015,         # HSV-Hue 扰动
        hsv_s=0.7,           # HSV-Saturation 扰动
        hsv_v=0.4,           # HSV-Value 扰动
        scale=0.5,           # 随机缩放
        fliplr=0.5,          # 水平翻转概率
        flipud=0.0,          # 不上下翻转（人物倒立不合理）
        # 训练超参
        batch=16,
        device=0,            # GPU 0；无 GPU 用 device="cpu"
        workers=4,
        # 保存与验证
        save=True,
        save_period=10,      # 每 10 个 epoch 存一次权重
        val=True,
        # 项目命名
        project=str(ROOT / "runs" / "train"),
        name="yolov8n_player",
        exist_ok=True,
        # 其他
        pretrained=True,
        optimizer="auto",
        verbose=True,
        seed=42,
    )

    # ── 导出最佳权重路径 ────────────────────────────────
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n训练完成！最佳权重: {best_pt}")
