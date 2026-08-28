# Exp6: 基于 SegNet 的街景分割实验

本实验在 `PyTorch` 下实现一个简化版 `SegNet`，用于完成 `CamVid` 街景语义分割任务，并输出以下评估指标：

- 像素准确率 `PA (Pixel Accuracy)`
- 平均像素准确率 `MPA (Mean Pixel Accuracy)`
- 平均交并比 `mIoU (Mean Intersection over Union)`
- 频权交并比 `fwIoU (Frequency Weighted IoU)`

默认训练配置面向 `GPU` 运行，并尽量将显存占用控制在 `16GB` 以内：

- 默认设备：`cuda`
- 默认 batch size：`2`
- 默认开启混合精度：`AMP`
- 默认启用温和的类别加权损失
- 默认启用更强的数据增强（随机缩放裁剪、翻转、亮度/对比度扰动）
- 若仍出现显存不足，可继续减小 `--batch_size`

实验目录结构参考了前面的实验模板，核心文件包括：

- `src/exp6/config.py`：默认路径与超参数
- `src/exp6/data_utils.py`：CamVid 数据读取与标签编码
- `src/exp6/model.py`：SegNet 模型
- `src/exp6/engine.py`：训练、验证、预测与断点保存
- `src/exp6/evaluator.py`：分割评估指标
- `src/exp6/main.py`：命令行入口

## 1. 数据集准备

本实验使用 `Cambridge-driving Labeled Video Database (CamVid)`。

本项目当前采用的可用数据来源：

- `GitCode` 镜像仓库：<https://gitcode.com/open-source-toolkit/1422a.git>

原始官方参考页面：

- CamVid 官方页面：<http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/>

推荐直接使用已经整理好的 `CamVid.zip`，并解压到 `data/` 目录下。

请将数据集整理为如下目录结构：

```text
data/CamVid/
├─ train/
├─ train_labels/
├─ val/
├─ val_labels/
├─ test/
└─ test_labels/
```

说明：

- `train/`、`val/`、`test/` 中存放原始街景图片。
- `train_labels/`、`val_labels/`、`test_labels/` 中存放对应标签图。
- 标签图既支持类别索引灰度图，也支持 CamVid 常见的 RGB 彩色标注图。
- 默认通过同名文件匹配图像与标签，例如 `0001TP_006690.png` 对应 `train_labels/0001TP_006690.png`。

### 1.1 为什么不再使用旧的数据获取方式

本实验早期尝试过直接使用 CamVid 官方页面提供的旧链接获取数据，但目前不再作为默认方案，原因如下：

- 官方给出的 `701_StillsRaw_full.zip` 原图临时链接已经失效，实际返回的是错误网页，而不是有效压缩包。
- 官方页面中的 FTP 视频入口当前可读到页面说明，但在本地环境中无法稳定解析和下载视频文件，因此不适合作为当前实验的默认数据准备方案。
- 即使改走“原始视频 + 手工提帧”路线，也会增加额外的数据准备复杂度，不利于本实验聚焦 `SegNet` 训练、评估与结果分析。

因此，当前 README 默认采用已经整理好的 `CamVid.zip` 数据包。这样可以直接获得：

- 已划分好的 `train / val / test`
- 与原图一一对应的标签目录
- 可直接用于本实验代码的数据结构

这也是当前仓库中推荐的数据准备方式。

## 2. 环境依赖

建议 Python `3.10+`。

安装依赖：

```powershell
pip install -r src/exp6/requirements.txt
```

如果使用 GPU，请先根据本机 CUDA 版本安装官方 `PyTorch`。

## 2.1 自动部署实验数据

如果你已经下载并解压了 CamVid，可以使用脚本自动整理数据：

```powershell
python src/exp6/prepare_camvid.py --raw_dir E:\你的CamVid原始目录 --output_dir E:\hw\deep_learning\data\CamVid
```

支持两类常见原始结构：

- 已经按 `train / val / test` 分好目录
- 原图与标签在两个大目录中，再配合 `train.txt / val.txt / test.txt`

如果你当前使用的是已经整理好的 `CamVid.zip`，通常不需要再执行旧的自动下载逻辑，直接解压到 `data/` 下即可。

如果你想直接自动下载 CamVid，也可以让脚本先下载再部署：

```powershell
python src/exp6/prepare_camvid.py --download --output_dir E:\hw\deep_learning\data\CamVid
```

注意：

- 脚本内部已经固定使用 CamVid 官方静态图与标签压缩包地址。
- 由于官方旧静态图链接目前并不稳定，这条自动下载路径仅保留为兼容方案，不再作为本实验默认推荐方式。
- 如果你已经有本地整理好的数据包，优先直接解压并使用。

更详细的用法见：

- `src/exp6/deploy_data.md`

## 2.2 从本地视频提帧

如果你拿到了 CamVid 原始视频文件，而不是 `701_StillsRaw_full.zip`，可以先根据标签名生成提帧清单：

```powershell
python src/exp6/extract_camvid_video_frames.py ^
  --labels_dir E:\hw\deep_learning\data\CamVid\labels_only ^
  --output_dir E:\hw\deep_learning\data\CamVid\raw_from_videos ^
  --manifest_csv E:\hw\deep_learning\data\CamVid\frame_manifest.csv
```

这一步默认只做 `dry-run`，不会真的提帧，但会输出：

- 每个序列需要提多少张图
- 标签名对应的原图名
- 全局帧号与局部帧号

如果你已经把视频放到某个本地目录，再执行真正提帧：

```powershell
python src/exp6/extract_camvid_video_frames.py ^
  --labels_dir E:\hw\deep_learning\data\CamVid\labels_only ^
  --videos_dir E:\CamVidVideos ^
  --output_dir E:\hw\deep_learning\data\CamVid\raw_from_videos ^
  --extract
```

默认视频文件名约定为：

- `01TP_extract.avi`
- `0005VD.MXF`
- `0006R0.MXF`
- `0016E5.MXF`

如果你的文件名不一样，可以手动覆盖，例如：

```powershell
python src/exp6/extract_camvid_video_frames.py ^
  --labels_dir E:\hw\deep_learning\data\CamVid\labels_only ^
  --output_dir E:\hw\deep_learning\data\CamVid\raw_from_videos ^
  --video_map 0001TP=E:\CamVidVideos\01TP_extract.avi ^
  --video_map Seq05VD=E:\CamVidVideos\0005VD.MXF ^
  --video_map 0006R0=E:\CamVidVideos\0006R0.MXF ^
  --video_map 0016E5=E:\CamVidVideos\0016E5.MXF ^
  --video_map 0016E5_15Hz=E:\CamVidVideos\0016E5.MXF ^
  --extract
```

注意：

- 我已经确认当前环境里 `ffmpeg` 可用。
- 但我没法替你直接下载官方 FTP 视频，因为该 FTP 主机当前无法解析。
- 所以这条方案现在依赖你先拿到本地视频文件。

## 3. 训练

在项目根目录 `E:\hw\deep_learning` 下运行：

```powershell
python src/exp6/main.py --mode train
```

默认即使用 `GPU` 训练；只有在本机不可用 CUDA 时，代码才会自动回退到 `CPU`。

常用训练命令：

```powershell
# 16GB 显存内更稳妥的默认推荐
python src/exp6/main.py --mode train --device cuda --batch_size 2

# 显存更宽裕时再尝试增大 batch size
python src/exp6/main.py --mode train --device cuda --batch_size 4 --epochs 80

# 指定输入尺寸
python src/exp6/main.py --mode train --image_height 360 --image_width 480

# 仅在没有可用 GPU 时再使用 CPU
python src/exp6/main.py --mode train --device cpu --no_amp

# 断点续训
python src/exp6/main.py --mode train --resume

# 从指定模型恢复
python src/exp6/main.py --mode train --resume --resume_ckpt src/exp6/checkpoints/segnet_camvid.last.pth
```

训练时会输出：

- `train_loss`
- 验证集 `PA / MPA / mIoU`
- 测试集 `PA / MPA / mIoU`

当前版本已针对 CamVid 小类和长尾类别做了两项默认优化：

- `Weighted CrossEntropy`
- 更强的数据增强以提升小目标与边界类的泛化

模型会保存到：

- `src/exp6/checkpoints/segnet_camvid.best.pth`：验证集 `mIoU` 最优模型
- `src/exp6/checkpoints/segnet_camvid.last.pth`：最近一次训练状态
- `src/exp6/checkpoints/segnet_camvid.pth`：与最优模型同步的兼容名称

训练日志会写入：

- `src/exp6/logs/*.jsonl`

## 4. 评估

使用最优模型在验证集和测试集上评估：

```powershell
python src/exp6/main.py --mode eval
```

指定模型评估：

```powershell
python src/exp6/main.py --mode eval --resume_ckpt src/exp6/checkpoints/segnet_camvid.best.pth
```

控制台将输出：

- `PA`
- `MPA`
- `mIoU`
- `fwIoU`
- 每个类别的 `IoU`

## 5. 预测与可视化

将测试集预测结果保存为彩色分割图：

```powershell
python src/exp6/main.py --mode predict
```

或指定输出目录：

```powershell
python src/exp6/main.py --mode predict --pred_dir src/exp6/predictions_demo
```

输出结果默认保存到：

- `src/exp6/predictions/`

### 5.1 训练日志与结果可视化

为了和前面实验的交付方式保持一致，`exp6` 额外提供了训练日志清洗、曲线绘制和测试示例图导出脚本：

```powershell
python src/exp6/visualize_results.py --log_file src/exp6/logs/你的训练日志.jsonl
```

如果不传 `--log_file`，脚本会默认选择 `src/exp6/logs/` 下最新的日志文件。

脚本会输出：

- `cleaned_log.jsonl`
- `epoch_loss.png`
- `miou_curve.png`
- `accuracy_curve.png`
- `summary.md`
- `examples/*.png`

其中测试示例图会按 `Input / GroundTruth / Prediction` 三列拼接，方便直接放进实验报告。

常用命令示例：

```powershell
python src/exp6/visualize_results.py --log_file src/exp6/logs/segnet_camvid__20260628_161601.jsonl --num_examples 6
```

## 6. 指标说明

设混淆矩阵为 `M`：

- `PA = sum(diag(M)) / sum(M)`
- `MPA = mean_i( Mii / sum_j(Mij) )`
- `IoU_i = Mii / (sum_j(Mij) + sum_j(Mji) - Mii)`
- `mIoU = mean_i(IoU_i)`

这些指标已经在 `src/exp6/evaluator.py` 中实现。

## 7. 实验报告可写内容建议

你可以在实验报告中按下面结构描述：

1. 实验目的
   使用 `SegNet` 完成 CamVid 街景语义分割任务。
2. 模型结构
   采用编码器-解码器结构，编码阶段使用 `MaxPool` 保存池化索引，解码阶段使用 `MaxUnpool` 恢复空间分辨率。
3. 数据处理
   对输入图像统一缩放，并将标签图映射为类别索引。
4. 损失函数
   使用 `CrossEntropyLoss`。
5. 评估指标
   使用 `PA`、`MPA`、`mIoU`。
6. 实验结果
   填入训练后控制台输出的验证集与测试集指标数值。

## 8. 注意事项

- 如果报错 `folder not found`，先检查 `data/CamVid/` 目录结构是否正确。
- 如果显存不足，可尝试：
  - 减小 `--batch_size`
  - 保持 `--use_amp`，不要关闭混合精度
  - 减小 `--image_height` 和 `--image_width`
  - 最后再考虑使用 `--device cpu`
- 如果标签颜色与标准 CamVid 颜色不一致，请在 `src/exp6/config.py` 中修改 `CAMVID_COLORS`。
