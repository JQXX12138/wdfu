# Delta Force — YOLOv8 Person Detection & Auto-Aim

基于 YOLOv8 的游戏人物检测与实时跟枪系统。

## 项目结构

```
├── scripts/
│   ├── train.py                # 训练脚本
│   ├── inference_image.py      # 图片推理 + 可视化
│   ├── inference_video.py      # 视频推理
│   ├── inference_camera.py     # 摄像头实时检测
│   ├── inference_track.py      # 检测 + ByteTrack 多目标跟踪
│   └── inference_screen.py     # 游戏截屏 + 跟枪
├── data.yaml                   # 数据集配置
└── models/                     # 模型权重 (本地保存，不传 git)
```

## 快速开始

```bash
# 安装依赖
pip install ultralytics opencv-python numpy mss

# 图片推理
python scripts/inference_image.py --source data/sample.jpg

# 摄像头实时检测
python scripts/inference_camera.py

# 游戏截屏 + 跟枪
python scripts/inference_screen.py --display-mode full --aim-key ralt
```

## 训练

数据集：三角洲行动游戏截图，3000+ 张，单类别 "player"。

```bash
python scripts/train.py
```

| 指标 | 数值 |
|------|------|
| mAP@50 | 0.667 |
| mAP@50-95 | 0.278 |
| 推理速度 | ~3ms (RTX 4060) |
| 模型大小 | 6.3MB |

## 环境

- Python 3.9
- PyTorch 2.5 + CUDA 12.1
- ultralytics 8.4
- Windows 11
