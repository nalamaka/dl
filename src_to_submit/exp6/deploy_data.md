# CamVid 数据自动部署

`src_to_submit/exp6/prepare_camvid.py` 用于把你本地已经下载好的 CamVid 原始数据整理到实验所需目录，并且支持指定部署路径。

现在它也支持：

- 从内置的 CamVid 官方静态图和标签链接自动下载压缩包
- 自动解压
- 解压后继续部署为实验目录结构

## 1. 标准用法

如果你的原始数据已经按 `train / val / test` 和对应标签目录拆好：

```powershell
python src_to_submit/exp6/prepare_camvid.py --raw_dir E:\dataset\CamVidRaw --output_dir E:\hw\deep_learning\data\CamVid
```

脚本会自动识别这些目录名：

- 图像目录：`train`、`val`、`test`
- 标签目录：`train_labels`、`val_labels`、`test_labels`

也兼容部分常见标签目录名：

- `trainannot`
- `valannot`
- `testannot`

## 1.1 下载后再部署

如果你想直接自动下载 CamVid 并部署，现在不需要再手动提供下载链接，只要指定数据位置即可：

```powershell
python src_to_submit/exp6/prepare_camvid.py --download --output_dir E:\hw\deep_learning\data\CamVid
```

说明：

- 脚本内置下载这两个官方文件：
  - `701_StillsRaw_full.zip`
  - `LabeledApproved_full.zip`
- 若压缩包解压后只有一层根目录，脚本会自动进入该目录继续识别数据结构。

如果只想下载并解压，不立即部署：

```powershell
python src_to_submit/exp6/prepare_camvid.py --download --output_dir E:\hw\deep_learning\data\CamVid --download_only
```

可选参数：

- `--extract_dir`：指定解压目录
- `--overwrite`：覆盖已存在的压缩包或解压目录内容

## 2. 原始大目录 + 划分清单

如果你的数据是这类结构：

```text
CamVidRaw/
├─ 701_StillsRaw_full/
├─ LabeledApproved_full/
├─ train.txt
├─ val.txt
└─ test.txt
```

可以直接执行：

```powershell
python src_to_submit/exp6/prepare_camvid.py --raw_dir E:\dataset\CamVidRaw --output_dir E:\hw\deep_learning\data\CamVid
```

脚本会自动识别：

- 图片目录：`701_StillsRaw_full`
- 标签目录：`LabeledApproved_full`
- 划分文件：`train.txt`、`val.txt`、`test.txt`

## 3. 显式指定路径

如果目录名不一样，可以手动指定：

```powershell
python src_to_submit/exp6/prepare_camvid.py ^
  --mode flat_with_txt ^
  --raw_dir E:\dataset\CamVidRaw ^
  --images_dir images ^
  --labels_dir masks ^
  --train_list split\train.txt ^
  --val_list split\val.txt ^
  --test_list split\test.txt ^
  --output_dir E:\hw\deep_learning\data\CamVid
```

## 4. 覆盖已有文件

如果目标目录已经存在文件，默认不会覆盖。需要覆盖时加：

```powershell
python src_to_submit/exp6/prepare_camvid.py --raw_dir E:\dataset\CamVidRaw --output_dir E:\hw\deep_learning\data\CamVid --overwrite
```

## 5. 部署结果

部署后目录会变成：

```text
E:\hw\deep_learning\data\CamVid\
├─ train\
├─ train_labels\
├─ val\
├─ val_labels\
├─ test\
└─ test_labels\
```
