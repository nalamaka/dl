# 实验2 - ViT CIFAR10

## 安装

```bash
pip install -r requirements.txt
```

## 手动运行

默认超参数与路径配置在 `config.py` 中管理。
默认从 `../../data/` 读取 CIFAR10 数据（项目级 `data` 目录）。

训练（示例参数）：

```bash
python main.py --epochs 20 --batch_size 128 --patch 4 --lr 0.001 --image_size 32
```

训练后可视化（请使用与训练相同的参数）：

```bash
python visualize_results.py --batch_size 128 --patch 4 --image_size 32
```

注意事项：

- 如果训练时修改了 `--patch` 或 `--image_size`，请在 `visualize_results.py` 中传入相同取值。

## 输出文件

- `log/log_vit_patch4.txt`: 文本日志
- `log/history.csv`: 结构化训练历史
- `checkpoint/*-ckpt.t7`: 最优模型权重
- `visualizations/training_curve.png`: 训练曲线图
- `visualizations/case_showcase.png`: 预测样例网格图
