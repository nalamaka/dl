# Exp6: 基于 SegNet 的 CamVid 语义分割

本实验在 PyTorch 下实现简化版 SegNet，用于 CamVid 街景语义分割。

评估指标：
- PA (Pixel Accuracy)
- MPA (Mean Pixel Accuracy)
- mIoU (Mean Intersection over Union)
- fwIoU (Frequency Weighted IoU)

## 0. 一键跑通（推荐）

在项目根目录执行一条命令即可完成：依赖安装 -> 1 epoch 冒烟训练 -> 评估 -> 预测 -> 可视化。

```powershell
powershell -ExecutionPolicy Bypass -File src_to_submit/exp6/one_click_smoke.ps1
```

或直接双击：
- `src_to_submit/exp6/run_one_click.bat`

前提：`data/CamVid` 已按标准目录准备好（见下文“数据准备”）。

脚本当前采用轻量策略：
- 若找到已有 checkpoint（优先 `src_to_submit/exp6/checkpoints`，其次 `src/exp6/checkpoints`），会直接执行评估与预测，不再重复训练。
- 若没有 checkpoint，才会回退到 1 epoch 冒烟训练后再评估与预测。

## 1. 目录与代码管理

`src_to_submit/exp6` 目录包含：
- `main.py`: 命令行入口（train/eval/predict）
- `config.py`: 路径与超参数配置
- `data_utils.py`: 数据读取、增强、标签编码/解码
- `model.py`: SegNet 模型定义
- `engine.py`: 训练、评估、推理与断点保存
- `evaluator.py`: 指标实现
- `visualize_results.py`: 日志清洗与图表导出
- `prepare_camvid.py`: CamVid 数据部署脚本
- `extract_camvid_video_frames.py`: 从视频提取帧的辅助脚本
- `inspect_camvid_labels.py`: 标签检查工具
- `doc/report.md`: 实验报告模板

运行中会生成：
- `checkpoints/`: 模型权重
- `logs/`: 训练日志（jsonl）
- `visualizations/`: 可视化结果

## 2. 数据准备

默认数据目录：
- `data/CamVid`

要求结构：

```text
project_root/
├─ data/
│  └─ CamVid/
│     ├─ train/
│     ├─ train_labels/
│     ├─ val/
│     ├─ val_labels/
│     ├─ test/
│     └─ test_labels/
└─ src_to_submit/
   └─ exp6/
```

如果你已经有 CamVid 原始包，可使用：

```powershell
python src_to_submit/exp6/prepare_camvid.py --raw_dir E:\你的CamVid原始目录 --output_dir E:\hw\deep_learning\data\CamVid
```

更多数据部署说明见：
- `src_to_submit/exp6/deploy_data.md`

## 3. 环境安装

建议 Python 3.10+。

```powershell
pip install -r src_to_submit/exp6/requirements.txt
```

如需可视化曲线，请额外安装：

```powershell
pip install matplotlib
```

如果 PowerShell 脚本执行被系统策略拦截，可先在当前终端执行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

## 4. 训练

在项目根目录执行：

```powershell
python src_to_submit/exp6/main.py --mode train
```

常用命令：

```powershell
# 16GB 显存推荐起步
python src_to_submit/exp6/main.py --mode train --device cuda --batch_size 2

# 指定输入分辨率
python src_to_submit/exp6/main.py --mode train --image_height 360 --image_width 480

# 断点续训
python src_to_submit/exp6/main.py --mode train --resume

# 指定恢复模型
python src_to_submit/exp6/main.py --mode train --resume --resume_ckpt src_to_submit/exp6/checkpoints/segnet_camvid.last.pth
```

训练输出：
- `src_to_submit/exp6/checkpoints/segnet_camvid.best.pth`
- `src_to_submit/exp6/checkpoints/segnet_camvid.last.pth`
- `src_to_submit/exp6/checkpoints/segnet_camvid.pth`
- `src_to_submit/exp6/logs/*.jsonl`

## 5. 评估

```powershell
python src_to_submit/exp6/main.py --mode eval
```

指定 checkpoint：

```powershell
python src_to_submit/exp6/main.py --mode eval --resume_ckpt src_to_submit/exp6/checkpoints/segnet_camvid.best.pth
```

## 6. 预测

```powershell
python src_to_submit/exp6/main.py --mode predict
```

指定输出目录：

```powershell
python src_to_submit/exp6/main.py --mode predict --pred_dir src_to_submit/exp6/visualizations/predictions_demo
```

## 7. 可视化

```powershell
python src_to_submit/exp6/visualize_results.py --log_file src_to_submit/exp6/logs/你的训练日志.jsonl --out_dir src_to_submit/exp6/visualizations/run1
```

典型输出：
- `cleaned_log.jsonl`
- `epoch_loss.png`
- `miou_curve.png`
- `accuracy_curve.png`
- `summary.md`
- `examples/*.png`

## 8. 快速自检

```powershell
python src_to_submit/exp6/main.py --mode train --epochs 1 --batch_size 2
python src_to_submit/exp6/main.py --mode eval --batch_size 2
```

一键自检版本（推荐）见：

```powershell
powershell -ExecutionPolicy Bypass -File src_to_submit/exp6/one_click_smoke.ps1
```

如果你只想“意思意思”快速过测，不想训练，可直接复用已有权重：

```powershell
python src_to_submit/exp6/main.py --mode eval --resume_ckpt src/exp6/checkpoints/segnet_camvid.best.pth
python src_to_submit/exp6/main.py --mode predict --resume_ckpt src/exp6/checkpoints/segnet_camvid.best.pth --pred_dir src_to_submit/exp6/visualizations/predictions_quick
```

## 9. 提交建议

建议提交：
- `src_to_submit/exp6/*.py`
- `src_to_submit/exp6/README.md`
- `src_to_submit/exp6/requirements.txt`
- `src_to_submit/exp6/deploy_data.md`
- `src_to_submit/exp6/doc/report.md`

通常不提交完整数据集；如课程要求复现实验结果，可附带训练好的 checkpoint。
